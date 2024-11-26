#pragma once

#include "tgaimage.hpp"
#include "model.hpp"

struct IShader {
    IShader() {}
    virtual ~IShader() {}

    Matrix4 transform;

    Matrix4 m_projection;
    Matrix4 m_view;
    Matrix4 m_viewport;

    Model *model;
    
    virtual Vec4f vertex(int face_index, int vert_index) = 0;
    virtual bool fragment(Vec3f bar, TGAColor &color) = 0;
};

namespace renderer {

    Matrix4 viewport(int w, int h);
    Matrix4 projection(float camera_distance);
    Matrix4 loot_at(Vec3f eye, Vec3f target = Vec3f(0,0,0), Vec3f up = Vec3f(0,1,0));

    /**
     * @brief Renders model to output image, according to shader's behaviour.
     * 
     * @return std::vector<float> z buffer created during rendering
     */
    std::vector<float> render(TGAImage &output, Model &model, IShader &shader);
}