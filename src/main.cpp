#include <cmath>
#include <utility>
#include <iostream>
#include <algorithm>
#include <array>

#include "tgaimage.hpp"
#include "model.hpp"
#include "renderer.hpp"


struct GouraudShader : IShader {
	float varying_intensity[3];
	Vec2f varying_uv[3];

    virtual Vec4f vertex(int face_index, int vert_index) {
		std::vector<int> face_vert_indexes = model->face(face_index);
		std::vector<int> face_uv_indexes = model->face_uvs(face_index);
		std::vector<int> face_normal_indexes = model->face_normals(face_index);

		varying_uv[vert_index] = model->uv(face_uv_indexes[vert_index]);

		Vec3f normal = model->normal(face_normal_indexes[vert_index]);
		varying_intensity[vert_index] = std::max(0.0f, normal.dot(Vec3f(0, 0, -1)));

		return transform * model->vert(face_vert_indexes[vert_index]).homogenize();
	}

    virtual bool fragment(Vec3f bar, TGAColor &color) {
		// find fragments's light intensity
		float light_intensity = varying_intensity[0] * bar.x + varying_intensity[1] * bar.y + varying_intensity[2] * bar.z;
		if (light_intensity <= 0) {
			return false;
		}

		// find the pixel's color by interpolating the diffuse texture
		Vec2f pixel_uv = varying_uv[0] * bar.x + varying_uv[1] * bar.y + varying_uv[2] * bar.z;
		color = model->sample_texture(pixel_uv) * light_intensity;

		return true;
	}
};

int main(int argc, char** argv) {
	TGAImage output(400, 400, TGAImage::RGB);
	
	Model model("models/african_head/african_head.obj");
	model.load_texture("models/african_head/african_head_diffuse.tga");

	GouraudShader shader;

	shader.m_projection = renderer::projection(5);
	shader.m_view = renderer::loot_at(Vec3f(1, 0, -2));
	
	renderer::render(output, model, shader);

	output.flip_vertically(); // so the origin is left bottom corner
	output.write_tga_file("build/out/img.tga");

	return 0;
}
