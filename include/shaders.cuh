#pragma once

#include "renderer.cuh"

struct PhongShaderData : IShaderData
{
    Vec3f light_dir = Vec3f(0.8f, 0.8f, 0.3f);

    Matrix4 PVM_it; // inverse transpose of projection * modelview
    Vec3f l;        // transformed light direction

    PhongShaderData()
    {
        m_projection = renderer::projection(3);
        m_view = renderer::loot_at(Vec3f(0.25f, 0.25f, 1.0f));

        Matrix4 PVM = m_projection * m_view;
        if (!PVM.transpose().inverse(PVM_it))
        {
            // fallback if matrix is not invertible
            PVM_it = Matrix4::identity();
        }

        l = PVM.mult(light_dir, false).normalize() * -1;
    }
};

struct PhongShader : IShader<PhongShaderData>
{
    Vec2f varying_uv[3];

    __device__ Vec4f vertex(int face_index, int vert_index)
    {
        int *face_vert_indexes = model->face(face_index);
        int *face_uv_indexes = model->face_uvs(face_index);

        varying_uv[vert_index] = model->uv(face_uv_indexes[vert_index]);
        
        return uniform->transform * model->vert(face_vert_indexes[vert_index]).homogenize();
    }

    __device__ bool fragment(Vec3f bar, TGAColor &color)
    {

        Vec2f uv = varying_uv[0] * bar.x + varying_uv[1] * bar.y + varying_uv[2] * bar.z;

        Vec3f n_sample = Vec3f::from_tgacolor(model->sample_texture(uv, TextureType::NORMAL_MAP));
        float spec_sample = model->sample_texture(uv, TextureType::SPECULAR).b;

        Vec3f n = uniform->PVM_it.mult(n_sample, false).normalize();
        Vec3f r = (n * (2 * n.dot(uniform->l)) - uniform->l).normalize();

        float specular = spec_sample >= 1 ? powf(max(r.z, 0.0f), spec_sample) : 0.0f;
        float diffuse = max(0.0f, n.dot(uniform->l));

        color = model->sample_texture(uv, TextureType::DIFFUSE);

        for (int i = 0; i < 3; i++)
        {
            color.raw[i] = min(color.raw[i] * (0.8f * diffuse + 0.6f * specular), 255.0f);
        }

        return true;
    }
};