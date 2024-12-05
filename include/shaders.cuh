#pragma once

#include "renderer.cuh"

struct PhongShader : IShader
{
    Vec3f light_dir = Vec3f(0.8f, 0.8f, 0.3f);

    Vec2f varying_uv[3];
    Matrix4 uniform_PVM_it; // inverse transpose of projection * modelview
    Vec3f uniform_l;        // transformed light direction

    PhongShader()
    {
        m_projection = renderer::projection(3);
        m_view = renderer::loot_at(Vec3f(0.25f, 0.25f, 1));

        Matrix4 uniform_PVM = m_projection * m_view;
        if (!uniform_PVM.transpose().inverse(uniform_PVM_it))
        {
            // fallback if matrix is not invertible
            uniform_PVM_it = Matrix4::identity();
        }

        uniform_l = uniform_PVM.mult(light_dir, false).normalize() * -1;
    }

    __device__ Vec4f vertex(int face_index, int vert_index)
    {
        int *face_vert_indexes = model->face(face_index);
        int *face_uv_indexes = model->face_uvs(face_index);

        varying_uv[vert_index] = model->uv(face_uv_indexes[vert_index]);
        return transform * model->vert(face_vert_indexes[vert_index]).homogenize();
    }

    __device__ bool fragment(Vec3f bar, TGAColor &color)
    {

        Vec2f uv = varying_uv[0] * bar.x + varying_uv[1] * bar.y + varying_uv[2] * bar.z;

        Vec3f n_sample = Vec3f::from_tgacolor(model->sample_texture(uv, TextureType::NORMAL_MAP));
        float spec_sample = model->sample_texture(uv, TextureType::SPECULAR).b;

        Vec3f n = uniform_PVM_it.mult(n_sample, false).normalize();
        Vec3f r = (n * (2 * n.dot(uniform_l)) - uniform_l).normalize();

        float specular = spec_sample >= 1 ? powf(max(r.z, 0.0f), spec_sample) : 0.0f;
        float diffuse = max(0.0f, n.dot(uniform_l));

        color = model->sample_texture(uv, TextureType::DIFFUSE);

        for (int i = 0; i < 3; i++)
        {
            color.raw[i] = min(color.raw[i] * (0.8f * diffuse + 0.6f * specular), 255.0f);
        }

        return true;
    }
};