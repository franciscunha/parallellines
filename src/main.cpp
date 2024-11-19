#include <cmath>
#include <utility>
#include <iostream>
#include <array>

#include "paths.hpp"
#include "tgaimage.hpp"
#include "model.hpp"

const TGAColor white = TGAColor(255, 255, 255, 255);
const TGAColor red   = TGAColor(255, 0,   0,   255);
const TGAColor blue  = TGAColor(0,   0,   255, 255);

const int width = 1024;
const int height = 1024;

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

void debug_line(int x0, int y0, int x1, int y1, TGAImage &image) {
	image.set(x0, y0, red);
	image.set(x1, y1, red);
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

int main(int argc, char** argv) {
	TGAImage image(width, height, TGAImage::RGB);

	Model 
	model = Model(".\\obj\\african_head.obj");	
	wireframe(model, image, white);

	image.flip_vertically(); // i want to have the origin at the left bottom corner of the image
	image.write_tga_file(paths::output("img.tga"));

	return 0;
}
