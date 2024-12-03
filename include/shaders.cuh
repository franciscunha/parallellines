#pragma once

#include "renderer.cuh"

struct PhongShader : IShader 
{
    Vec3f light_dir;

	Vec2f varying_uv[3];
	Matrix4 uniform_PVM_it; // inverse transpose of projection * modelview
	Vec3f uniform_l;		// transformed light direction

    
    __device__ Vec4f vertex(int face_index, int vert_index);
    __device__ bool fragment(Vec3f bar, TGAColor &color);
};