#pragma once

#include "tgaimage.cuh"
#include "model.cuh"

struct IShader
{
    IShader() {}
    virtual ~IShader() {}

    Matrix4 transform;

    Matrix4 m_projection;
    Matrix4 m_view;
    Matrix4 m_viewport;

    Model *model;

    __device__ Vec4f vertex(int face_index, int vert_index) { return Vec4f(0, 0, 0, 0); };
    __device__ bool fragment(Vec3f bar, TGAColor &color) { return false; };
};

namespace renderer
{

    Matrix4 viewport(int w, int h);
    Matrix4 projection(float camera_distance);
    Matrix4 loot_at(Vec3f eye, Vec3f target = Vec3f(0, 0, 0), Vec3f up = Vec3f(0, 1, 0));

    /**
     * @brief Renders model to output image, according to shader's behaviour.
     */
    void render(TGAImage &output, Model &model, IShader &shader);
}