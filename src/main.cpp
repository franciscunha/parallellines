#include <cmath>
#include <utility>
#include <iostream>
#include <algorithm>
#include <array>

#include "tgaimage.hpp"
#include "model.hpp"
#include "renderer.hpp"


int main(int argc, char** argv) {
	TGAImage output(400, 400, TGAImage::RGB);
	
	Model model("models/african_head/african_head.obj");
	model.load_texture("models/african_head/african_head_diffuse.tga");

	Renderer r(output, model);
	
	r.set_camera_distance(5);
	r.set_light_dir(Vec3f(0, 0, -1));
	r.render();

	output.flip_vertically(); // so the origin is left bottom corner
	output.write_tga_file("build/out/img.tga");

	return 0;
}
