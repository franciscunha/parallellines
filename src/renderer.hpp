#pragma once

#include "tgaimage.hpp"
#include "model.hpp"

class Renderer {
private:
    TGAImage &output;
    Model &model;

    const int width;
    const int height;
    std::vector<float> z_buffer;
    Vec3f light_dir;

    Matrix4 transform;

    Matrix4 m_model;
    Matrix4 m_projection;
    Matrix4 m_view;
    Matrix4 m_viewport;

    Vec3f barycentric(Vec2i triangle[3], Vec2i p);
    void render_face(int face_index);

public:
    Renderer(TGAImage &output, Model &model);

    void set_light_dir(Vec3f dir);
    void set_camera_distance(float c);
    void loot_at(Vec3f eye, Vec3f target = Vec3f(0,0,0), Vec3f up = Vec3f(0,1,0));

    void render();
};