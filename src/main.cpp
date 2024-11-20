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

const int width = 256;
const int height = 256;

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


int main(int argc, char** argv) {
	TGAImage image(width, height, TGAImage::RGB);

	Vec2i t0[3] = {Vec2i(10, 70),   Vec2i(50, 160),  Vec2i(70, 80)}; 
	Vec2i t1[3] = {Vec2i(180, 50),  Vec2i(150, 1),   Vec2i(70, 180)}; 
	Vec2i t2[3] = {Vec2i(180, 150), Vec2i(120, 160), Vec2i(130, 180)}; 
	triangle(t0[0], t0[1], t0[2], image, blue); 
	triangle(t1[0], t1[1], t1[2], image, white); 
	triangle(t2[0], t2[1], t2[2], image, green);

	image.flip_vertically(); // i want to have the origin at the left bottom corner of the image
	image.write_tga_file(paths::output("img.tga"));

	return 0;
}
