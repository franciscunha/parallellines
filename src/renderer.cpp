
#include <cmath>
#include <utility>
#include <iostream>
#include <algorithm>
#include <array>

#include "renderer.hpp"


Renderer::Renderer(TGAImage &output_, Model &model_) : 
    output(output_),
    model(model_),
    width(output_.get_width()),
    height(output_.get_height()),
    z_buffer(width * height, -std::numeric_limits<float>::max()),
	light_dir(Vec3f(0, 0, -1))
{
	// calculate viewport matrix
	
	constexpr int Z_DEPTH = 255;
	// scaling part 
	m_viewport.m[0][0] = (width   / 2.0f);
	m_viewport.m[1][1] = (height  / 2.0f);
	m_viewport.m[2][2] = (Z_DEPTH / 2.0f);
	// translating part
	for (int i = 0; i < 3; i++) {
		m_viewport.m[i][3] = m_viewport.m[i][i];
	}
    
	// other matrices are initialized to identity and may be changed through method calls
}

/**
 * Given three vertices of a triangle and a point P, returns the barycentric coordinates of P in relation to the triangle.
 * If triangle is degenerate, returns a negative coordinate in X.
 */
Vec3f Renderer::barycentric(Vec2i triangle[3], Vec2i p) {
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

void Renderer::render_face(int face_index) { 	
	// get face vertices and uv
	std::vector<int> face_vert_indexes = model.face(face_index);
	std::vector<int> face_uv_indexes = model.face_uvs(face_index);
	std::vector<int> face_normal_indexes = model.face_normals(face_index);

	Vec3f vertices[3];
	Vec2f uv[3];
	Vec3f normals[3];
	for (int j = 0; j < 3; j++) {
		// also perform all transformations on each vertex
		vertices[j] = (transform * model.vert(face_vert_indexes[j]).homogenize()).dehomogenize();
		uv[j] 		= model.uv(face_uv_indexes[j]);
		normals[j]  = model.normal(face_normal_indexes[j]);
	}

	// calculate the face's normal, and use that for backface culling
	Vec3f normal = ((vertices[1] - vertices[0]).cross(vertices[2] - vertices[0]));	
	Vec3f model_to_camera_dir = Vec3f(0, 0, 1);
	if (normal.dot(model_to_camera_dir) < 0) {
		return;
	}

	// round vertices to pixel coords
	Vec2i screen_coords[3];
	for (int i = 0; i < 3; i++) {
		screen_coords[i] = Vec2i(std::round(vertices[i].x), std::round(vertices[i].y));
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

	// pre-compute this pixel-independent value, used later for shading
	Vec3f reverse_light_dir = (light_dir * (-1)).normalize();

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

			if (z_buffer[(x * width) + y] > pixel_z) {
				// there is already something drawn in front of this pixel
				continue;
			}
			z_buffer[(x * width) + y] = pixel_z;

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

void Renderer::loot_at(Vec3f eye, Vec3f target, Vec3f up) {
	Vec3f z = (eye - target).normalize();
	Vec3f x = up.cross(z).normalize();
	Vec3f y = z.cross(x).normalize();

	Matrix4 translation = Matrix4::identity();
	Matrix4 inv_basis = Matrix4::identity();

	for (int i = 0; i < 3; i++) {
		translation.m[i][3] = -eye.raw[i];

		inv_basis.m[0][i] = x.raw[i];
		inv_basis.m[1][i] = y.raw[i];
		inv_basis.m[2][i] = z.raw[i];
	}

	m_view = inv_basis * translation;
}

void Renderer::set_light_dir(Vec3f dir) {
    light_dir = dir;
}

void Renderer::set_camera_distance(float c) {
    m_projection.m[3][2] = -(1/c);
}

void Renderer::render() {
	transform = m_viewport * m_projection * m_view * m_model;

	for (int i = 0; i < model.nfaces(); i++) { 
		render_face(i);
	}
}
