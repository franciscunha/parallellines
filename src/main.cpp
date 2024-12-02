#include <cmath>
#include <utility>
#include <iostream>
#include <algorithm>
#include <array>

#include "../include/tgaimage.hpp"
#include "../include/model.hpp"
#include "../include/renderer.hpp"

struct PhongShader : IShader
{
	Vec3f light_dir = Vec3f(0.8f, 0.8f, 0.3f);

	Vec2f varying_uv[3];
	Matrix4 uniform_PVM_it; // inverse transpose of projection * modelview
	Vec3f uniform_l;		// transformed light direction

	virtual Vec4f vertex(int face_index, int vert_index)
	{
		std::vector<int> face_vert_indexes = model->face(face_index);
		std::vector<int> face_uv_indexes = model->face_uvs(face_index);

		varying_uv[vert_index] = model->uv(face_uv_indexes[vert_index]);
		return transform * model->vert(face_vert_indexes[vert_index]).homogenize();
	}

	virtual bool fragment(Vec3f bar, TGAColor &color)
	{

		Vec2f uv = varying_uv[0] * bar.x + varying_uv[1] * bar.y + varying_uv[2] * bar.z;

		Vec3f n_sample = Vec3f::from_tgacolor(model->sample_texture(uv, TextureType::NORMAL_MAP));
		float spec_sample = model->sample_texture(uv, TextureType::SPECULAR).b;

		Vec3f n = uniform_PVM_it.mult(n_sample, false).normalize();
		Vec3f r = (n * (2 * n.dot(uniform_l)) - uniform_l).normalize();

		float specular = spec_sample >= 1 ? powf(std::max(r.z, 0.0f), spec_sample) : 0.0f;
		float diffuse = std::max(0.0f, n.dot(uniform_l));

		color = model->sample_texture(uv, TextureType::DIFFUSE);

		for (int i = 0; i < 3; i++)
		{
			color.raw[i] = std::min(color.raw[i] * (0.8f * diffuse + 0.6f * specular), 255.0f);
		}

		return true;
	}
};

int main(int argc, char **argv)
{
	TGAImage output(1024, 1024, TGAImage::RGB);

	Model model("models/african_head/african_head.obj");
	model.load_texture("models/african_head/african_head_diffuse.tga", TextureType::DIFFUSE);
	model.load_texture("models/african_head/african_head_spec.tga", TextureType::SPECULAR);
	model.load_texture("models/african_head/african_head_nm.tga", TextureType::NORMAL_MAP);

	PhongShader shader;

	shader.m_projection = renderer::projection(3);
	shader.m_view = renderer::loot_at(Vec3f(0.25f, 0.25f, 1));

	Matrix4 uniform_PVM = shader.m_projection * shader.m_view;
	if (!uniform_PVM.transpose().inverse(shader.uniform_PVM_it))
	{
		shader.uniform_PVM_it = Matrix4::identity();
	}
	shader.uniform_l = uniform_PVM.mult(shader.light_dir, false).normalize() * -1;

	renderer::render(output, model, shader);

	output.flip_vertically(); // so the origin is left bottom corner
	output.write_tga_file("images/out.tga");

	return 0;
}
