#include <cmath>
#include <utility>
#include <iostream>
#include <algorithm>
#include <array>

#include "tgaimage.hpp"
#include "model.hpp"
#include "renderer.hpp"


struct GouraudShader : IShader {
	Vec3f light_dir = Vec3f(0, 0, 1);

	float varying_intensity[3];
	Vec2f varying_uv[3];

    virtual Vec4f vertex(int face_index, int vert_index) {
		std::vector<int> face_vert_indexes = model->face(face_index);
		std::vector<int> face_uv_indexes = model->face_uvs(face_index);
		std::vector<int> face_normal_indexes = model->face_normals(face_index);

		varying_uv[vert_index] = model->uv(face_uv_indexes[vert_index]);

		Vec3f normal = model->normal(face_normal_indexes[vert_index]);
		varying_intensity[vert_index] = std::max(0.0f, normal.dot(light_dir));

		return transform * model->vert(face_vert_indexes[vert_index]).homogenize();
	}

    virtual bool fragment(Vec3f bar, TGAColor &color) {
		// find fragments's light intensity
		float light_intensity = varying_intensity[0] * bar.x + varying_intensity[1] * bar.y + varying_intensity[2] * bar.z;
		if (light_intensity <= 0) {
			return false;
		}

		// find the fragments's color by interpolating the diffuse texture
		Vec2f uv = varying_uv[0] * bar.x + varying_uv[1] * bar.y + varying_uv[2] * bar.z;
		color = model->sample_texture(uv, TextureType::DIFFUSE) * light_intensity;

		return true;
	}
};

struct NormalMappingShader : IShader {
	Vec3f light_dir = Vec3f(1, 0, 0);

	Vec2f varying_uv[3];
	Matrix4 uniform_PVM_it; // inverse transpose of projection * modelview
	Vec3f uniform_l; // transformed light direction

    virtual Vec4f vertex(int face_index, int vert_index) {
		std::vector<int> face_vert_indexes = model->face(face_index);
		std::vector<int> face_uv_indexes = model->face_uvs(face_index);

		varying_uv[vert_index] = model->uv(face_uv_indexes[vert_index]);
		return transform * model->vert(face_vert_indexes[vert_index]).homogenize();
	}

    virtual bool fragment(Vec3f bar, TGAColor &color) {

		// find fragments's UV
		Vec2f uv = varying_uv[0] * bar.x + varying_uv[1] * bar.y + varying_uv[2] * bar.z;

		// find fragment's normal through normal map, then transform it
		Vec3f sample = Vec3f::from_tgacolor(model->sample_texture(uv, TextureType::NORMAL_MAP)).normalize();
		Vec3f normal = (uniform_PVM_it * sample.homogenize(false)).dehomogenize().normalize();
		
		float light_intensity = normal.dot(uniform_l.normalize());
		if (light_intensity <= 0) {
			return false;
		}

		color = model->sample_texture(uv, TextureType::DIFFUSE) * light_intensity;

		return true;
	}
};


int main(int argc, char** argv) {
	TGAImage output(1024, 1024, TGAImage::RGB);
	
	Model model("models/african_head/african_head.obj");
	model.load_texture("models/african_head/african_head_diffuse.tga", TextureType::DIFFUSE);
	model.load_texture("models/african_head/african_head_nm.tga", TextureType::NORMAL_MAP);

	NormalMappingShader shader;

	shader.m_projection = renderer::projection(5);
	shader.m_view = renderer::loot_at(Vec3f(0.25f, 0.25f, 1));
	
	Matrix4 uniform_PVM = shader.m_projection * shader.m_view;
	if (!uniform_PVM.transpose().inverse(shader.uniform_PVM_it)) {
		shader.uniform_PVM_it = Matrix4::identity();
	}	
	shader.uniform_l = (uniform_PVM * shader.light_dir.homogenize(false)).dehomogenize().normalize() * -1;

	
	renderer::render(output, model, shader);

	output.flip_vertically(); // so the origin is left bottom corner
	output.write_tga_file("build/out/img.tga");

	return 0;
}
