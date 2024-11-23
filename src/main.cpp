#include <cmath>
#include <utility>
#include <iostream>
#include <algorithm>
#include <array>

#include "paths.hpp"
#include "tgaimage.hpp"
#include "model.hpp"

const TGAColor white  = TGAColor(255, 255, 255, 255);
const TGAColor red    = TGAColor(255, 0,   0,   255);
const TGAColor green  = TGAColor(0,   255, 0,   255);
const TGAColor blue   = TGAColor(0,   0,   255, 255);

const int width = 400;
const int height = 400;

float lerp(float v0, float v1, float t) {
  return (1 - t) * v0 + t * v1;
}

void line(int x0, int y0, int x1, int y1, TGAImage &image, TGAColor color) {
	
	bool steep = false;
	if (std::abs(x0 - x1) < std::abs(y0 - y1)) {
		steep = true;
		std::swap(x0, y0);
		std::swap(x1, y1);
	}
	
	if (x1 < x0) {
		std::swap(x0, x1);
		std::swap(y0, y1);
	}

	float slope = (y0 - y1) / (float)(x0 - x1);
	float y = y0;

	for (int x = x0; x <= x1; x++) {
		if (!steep) {
			image.set(x, std::round(y), color);
		} else {
			image.set(std::round(y), x, color);
		}
		y = y + slope;
	}
}

void line(Vec2i p0, Vec2i p1, TGAImage &image, TGAColor color) {
	line(p0.x, p0.y, p1.x, p1.y, image, color);
}

Vec2i world_to_screen_coords(Vec3f world_coords) {
	return Vec2i(
		std::round((world_coords.x + 1.0f) * (float)width  / 2.0f),
		std::round((world_coords.y + 1.0f) * (float)height / 2.0f)
	); 
}

void wireframe(Model &model, TGAImage &image, TGAColor color) {

	for (int i = 0; i < model.nfaces(); i++) {
		std::vector<int> face_vertices = model.face(i);

		for (int j = 0; j < 3; j++) {
			Vec2i vertex0 = world_to_screen_coords(model.vert(face_vertices[j]));
			Vec2i vertex1 = world_to_screen_coords(model.vert(face_vertices[(j+1) % 3]));
			line( vertex0.x, vertex0.y, vertex1.x, vertex1.y, image, color);
		}
		
	}
	
}

/**
 * Given three vertices of a triangle and a point P, returns the barycentric coordinates of P in relation to the triangle.
 * If triangle is degenerate, returns a negative coordinate in X.
 */
Vec3f barycentric(Vec2i triangle[3], Vec2i p) {
	Vec2i vAB = triangle[1] - triangle[0];
	Vec2i vAC = triangle[2] - triangle[0];
	Vec2i vPA = triangle[0] - p;

	Vec3f cross /* (u*w, v*w, w) */ = Vec3f(vAB.x, vAC.x, vPA.x).cross(Vec3f(vAB.y, vAC.y, vPA.y));
	
	float w = cross.z;
	if (std::abs(w) < std::numeric_limits<float>::epsilon()) {
		// w is actually the triangle's area, such that w == 0 indicates a degenerate triangle
		// in which case return a negative coordinate so that it is discarded
		return Vec3f(-1, 1, 0);
	}
	
	float u = cross.x / w;
	float v = cross.y / w;

	// even though its mathematically equivalent and more readable, 1.0f - u - v gives rounding
	// errors that lead to missing pixels on triangle edges. 1.0f - (cross.x+cross.y) / w somehow
	// avoids this.
	return Vec3f(1.0f - (cross.x+cross.y) / w, u, v); 
}

void render_face(
	Model &model,
	int face_index,
	Matrix4 &transform,
	Vec3f light_dir,
	float z_buffer[width][height],
	TGAImage &output
) { 	
	// get face vertices and uv
	std::vector<int> face_vert_indexes = model.face(face_index);
	std::vector<int> face_uv_indexes = model.face_uvs(face_index);
	std::vector<int> face_normal_indexes = model.face_normals(face_index);

	Vec3f vertices[3];
	Vec2f uv[3];
	Vec3f normals[3];
	for (int j = 0; j < 3; j++) {
		vertices[j] = (transform * model.vert(face_vert_indexes[j]).homogenize()).dehomogenize();
		uv[j] 		= model.uv(face_uv_indexes[j]);
		normals[j]  = model.normal(face_normal_indexes[j]);
	}

	Vec3f reverse_light_dir = (light_dir * (-1)).normalize();
	
	// calculate the face's normal, and use that for backface culling
	Vec3f normal = ((vertices[1] - vertices[0]).cross(vertices[2] - vertices[0])).normalize();	
	if (normal.dot(reverse_light_dir) < 0) {
		return;
	}

	// convert vertex from world coords to screen coords
	Vec2i screen_coords[3];
	for (int i = 0; i < 3; i++) {
		screen_coords[i] = world_to_screen_coords(vertices[i]);
	}


	// find the triangle's bounding box (bb)
	Vec2i bounds[2] = {
		Vec2i(
			std::min({screen_coords[0].x, screen_coords[1].x, screen_coords[2].x}), 
			std::min({screen_coords[0].y, screen_coords[1].y, screen_coords[2].y})
		),
		Vec2i(
			std::max({screen_coords[0].x, screen_coords[1].x, screen_coords[2].x}), 
			std::max({screen_coords[0].y, screen_coords[1].y, screen_coords[2].y})
		)
	};

	// iterate over pixels contained in bb, to check if each is in triangle
	for (int x = bounds[0].x; x <= bounds[1].x; x++) {
		for (int y = bounds[0].y; y <= bounds[1].y; y++) {
			
			// check if pixel is inside triangle using barycentic coords
			Vec3f barycentric_coords = barycentric(screen_coords, Vec2i(x, y));
			
			if (barycentric_coords.x < 0 || barycentric_coords.y < 0 || barycentric_coords.z < 0) {
				// pixel is not inside triangle, also ignores degenerate triangles
				continue;
			}

			// we want the pixel's z in world space - to know if one is in front of another
			float pixel_z = 
				vertices[0].z * barycentric_coords.x + 
				vertices[1].z * barycentric_coords.y + 
				vertices[2].z * barycentric_coords.z;

			if (z_buffer[x][y] > pixel_z) {
				// there is already something drawn in front of this pixel
				continue;
			}
			z_buffer[x][y] = pixel_z;

			// find the pixel's color by interpolating the diffuse texture
			Vec2f pixel_uv = 
				uv[0] * barycentric_coords.x + 
				uv[1] * barycentric_coords.y + 
				uv[2] * barycentric_coords.z;
			TGAColor color = model.sample_texture(pixel_uv);

			// (Gouraud shading) find pixel's light intensity
			Vec3f pixel_normal = 
				normals[0] * barycentric_coords.x + 
				normals[1] * barycentric_coords.y + 
				normals[2] * barycentric_coords.z;
			float light_intensity = reverse_light_dir.dot(pixel_normal);
			if (light_intensity <= 0) {
				continue;
			}

			output.set(x, y, color * light_intensity);
		}	
	}
}

void render_model(Model &model, Vec3f light_dir, float camera_pos, TGAImage &output_image) {
	// initialize z-buffer to negative infinity
	float z_buffer[width][height];
	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {
			z_buffer[x][y] = -std::numeric_limits<float>::max();
		}
	}

	// calculate transform that will be used on each face (so far only projection)
	Matrix4 transform = Matrix4::projection(camera_pos);

	for (int i = 0; i < model.nfaces(); i++) { 
		render_face(model, i, transform, light_dir, z_buffer, output_image);
	}
	// render_face(model, 0, transform, light_dir, z_buffer, output_image);
}


int main(int argc, char** argv) {
	TGAImage output(width, height, TGAImage::RGB);
	
	Model model(".\\models\\african_head\\african_head.obj");
	model.load_texture(".\\models\\african_head\\african_head_diffuse.tga");
	
	render_model(model, Vec3f(0, 0, -1), 5, output);

	output.flip_vertically(); // so the origin is left bottom corner
	output.write_tga_file(paths::output("img.tga"));

	return 0;
}
