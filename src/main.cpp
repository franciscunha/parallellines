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

const int width = 512;
const int height = 512;

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

void wireframe(Model &model, TGAImage &image, TGAColor color) {

	for (int i = 0; i < model.nfaces(); i++) {
		std::vector<int> face_vertices = model.face(i);

		for (int j = 0; j < 3; j++) {
			Vec3f vertex0 = model.vert(face_vertices[j]);
			Vec3f vertex1 = model.vert(face_vertices[(j+1) % 3]);
		
			line(
				(vertex0.x + 1) * width  / (float)2,
				(vertex0.y + 1) * height / (float)2,
				(vertex1.x + 1) * width  / (float)2,
				(vertex1.y + 1) * height / (float)2,
				image,
				color
			);
		}
		
	}
	
}

void triangle(Vec2i t0, Vec2i t1, Vec2i t2, TGAImage &image, TGAColor color) { 	
	// find the triangle's bounding box (bb)
	Vec2i bounds[2] = {
		Vec2i(std::min({t0.x, t1.x, t2.x}), std::min({t0.y, t1.y, t2.y})),
		Vec2i(std::max({t0.x, t1.x, t2.x}), std::max({t0.y, t1.y, t2.y}))
	};
	
	// iterate over pixels contained in bb, to check if each is in triangle

	// pre compute some stuff that is pixel independent and needed for barycentric coord
	Vec2i ab = t1 - t0;
	Vec2i ac = t2 - t0;

	for (int x = bounds[0].x; x <= bounds[1].x; x++) {
		for (int y = bounds[0].y; y <= bounds[1].y; y++) {
			
			// check if pixel is inside triangle using barycentic coords

			Vec2i pa = t0 - Vec2i(x, y);
			Vec3f cross = Vec3f(ac.x, ab.x, pa.x) ^ Vec3f(ac.y, ab.y, pa.y);
			// deal with division by zero (degenerate triangle)
			if (std::abs(cross.z) < 0.1) {
				continue;
			}
			cross = cross * (1/cross.z);
			Vec3f barycentric = Vec3f(1 - cross.x - cross.y, cross.y, cross.x);
			
			if (barycentric.x < 0 || barycentric.y < 0 || barycentric.z < 0) {
				// pixel is not inside triangle
				continue;
			}

			image.set(x, y, color);
		}	
	}

}

Vec2i world_to_screen_coords(Vec3f world_coords) {
	return Vec2i(
		(world_coords.x + 1) * (float)width  / (float)2,
		(world_coords.y + 1) * (float)height / (float)2
	); 
}

TGAColor mult_color(const TGAColor &c, float x) {
	return TGAColor(c.r * x, c.g * x, c.b * x, c.a);
}

void render(Model &model, TGAImage &image, Vec3f light_dir) {
	for (int i = 0; i < model.nfaces(); i++) { 
		std::vector<int> face = model.face(i);
		
		// get face vertices in both coordinate systems
		Vec3f world_coords[3];
		Vec2i screen_coords[3]; 
		for (int j = 0; j < 3; j++) {
			world_coords[j]  = model.vert(face[j]);
			screen_coords[j] = world_to_screen_coords(world_coords[j]);
		}
		
		// calculate the face's normal, and use that to get rough lighting
		Vec3f normal = ((world_coords[2] - world_coords[0]) ^ (world_coords[1] - world_coords[0])).normalize();		
		float light_intensity = normal * light_dir.normalize(); // dot product

		// backface culling
		if (light_intensity < 0) {
			continue;
		}
		
		triangle(screen_coords[0], screen_coords[1], screen_coords[2], image, mult_color(white, light_intensity));
	}
}


int main(int argc, char** argv) {
	TGAImage image(width, height, TGAImage::RGB);
	Model model(".\\obj\\african_head.obj");
	
	render(model, image, Vec3f(0, 0, -1));

	image.flip_vertically(); // i want to have the origin at the left bottom corner of the image
	image.write_tga_file(paths::output("img.tga"));

	return 0;
}
