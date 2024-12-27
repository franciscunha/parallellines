// This file is forked from an initial version by Dmitry V. Sokolov (https://github.com/ssloy) that was
// provided as a starting point for the tinyrenderer course (https://github.com/ssloy/tinyrenderer)

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include "../include/model.cuh"

Model::Model(const char *filename)
{
    std::ifstream in;
    in.open(filename, std::ifstream::in);
    if (in.fail())
    {
        std::cerr << "failed to open " << filename << std::endl;
        return;
    }

    std::vector<std::vector<int>> faces;
    std::vector<std::vector<int>> faces_uvs;
    std::vector<std::vector<int>> faces_normals;
    std::vector<Vec3f> verts;
    std::vector<Vec3f> uvs;
    std::vector<Vec3f> normals;

    std::string line;
    while (!in.eof())
    {
        std::getline(in, line);
        std::istringstream iss(line.c_str());

        char trash;
        if (!line.compare(0, 2, "v "))
        {
            iss >> trash;
            Vec3f v;
            for (int i = 0; i < 3; i++)
                iss >> v.raw[i];
            verts.push_back(v);
        }
        else if (!line.compare(0, 3, "vt "))
        {
            iss >> trash >> trash;
            Vec3f vt;
            vt.raw[2] = 0;
            for (int i = 0; i < 2; i++)
                iss >> vt.raw[i];
            uvs.push_back(vt);
        }
        else if (!line.compare(0, 3, "vn "))
        {
            iss >> trash >> trash;
            Vec3f vn;
            for (int i = 0; i < 3; i++)
                iss >> vn.raw[i];
            normals.push_back(vn);
        }
        else if (!line.compare(0, 2, "f "))
        {
            std::vector<int> v_indexes;
            std::vector<int> t_indexes;
            std::vector<int> n_indexes;
            int v_idx, t_idx, n_idx;
            iss >> trash;
            while (iss >> v_idx >> trash >> t_idx >> trash >> n_idx)
            {
                // in wavefront obj all indices start at 1, not zero
                v_idx--;
                t_idx--;
                n_idx--;

                v_indexes.push_back(v_idx);
                t_indexes.push_back(t_idx);
                n_indexes.push_back(n_idx);
            }
            faces.push_back(v_indexes);
            faces_uvs.push_back(t_indexes);
            faces_normals.push_back(n_indexes);
        }
    }

    n_verts_ = verts.size();
    n_uvs_ = uvs.size();
    n_normals_ = normals.size();

    vectors_ = new Vec3f[n_verts_ + n_uvs_ + n_normals_];
    std::copy(verts.begin(), verts.end(), vectors_);
    std::copy(uvs.begin(), uvs.end(), vectors_ + n_verts_);
    std::copy(normals.begin(), normals.end(), vectors_ + n_verts_ + n_uvs_);

    n_faces_ = faces.size();

    indexes_ = new int[n_faces_ * 9];
    for (int i = 0; i < n_faces_; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            indexes_[(0 * 3 * n_faces_) + (i * 3 + j)] = faces[i][j];
            indexes_[(1 * 3 * n_faces_) + (i * 3 + j)] = faces_uvs[i][j];
            indexes_[(2 * 3 * n_faces_) + (i * 3 + j)] = faces_normals[i][j];
        }
    }

    diffuse_ = new TGAImage();
    normal_map_ = new TGAImage();
    specular_ = new TGAImage();
}

Model::~Model()
{
    delete[] indexes_;
    delete[] vectors_;

    delete diffuse_;
    delete normal_map_;
    delete specular_;
}

Model *Model::cudaDeepCopyToDevice()
{
    // make device copies of the data in this

    int *d_indexes;
    Vec3f *d_vectors;
    TGAImage *d_diffuse;
    TGAImage *d_specular;
    TGAImage *d_normal_map;

    int size_indexes = (n_faces_ * 9) * sizeof(int);
    int size_vectors = (n_verts_ + n_uvs_ + n_normals_) * sizeof(Vec3f);

    cudaMalloc(&d_indexes, size_indexes);
    cudaMalloc(&d_vectors, size_vectors);

    cudaMemcpy(d_indexes, indexes_, size_indexes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vectors, vectors_, size_vectors, cudaMemcpyHostToDevice);

    // textures need deep copies
    if (diffuse_)
        d_diffuse = diffuse_->cudaDeepCopyToDevice();
    if (specular_)
        d_specular = specular_->cudaDeepCopyToDevice();
    if (normal_map_)
        d_normal_map = normal_map_->cudaDeepCopyToDevice();

    // temporarily make this point to the device data s.t. device model points to them

    int *indexes = indexes_;
    Vec3f *vectors = vectors_;
    TGAImage *diffuse = diffuse_;
    TGAImage *specular = specular_;
    TGAImage *normal_map = normal_map_;

    indexes_ = d_indexes;
    vectors_ = d_vectors;
    diffuse_ = d_diffuse;
    specular_ = d_specular;
    normal_map_ = d_normal_map;

    // copy this to device

    Model *d_model;
    cudaMalloc(&d_model, sizeof(Model));
    cudaMemcpy(d_model, this, sizeof(Model), cudaMemcpyHostToDevice);

    // reset the host model pointers to the host arrays

    indexes_ = indexes;
    vectors_ = vectors;
    diffuse_ = diffuse;
    specular_ = specular;
    normal_map_ = normal_map;

    // return pointer to device model
    return d_model;
}

void Model::cudaDeepFree(Model *device_ptr)
{
    // copy device data to host so we can access the inner addresses
    Model *m = (Model *)malloc(sizeof(Model));
    cudaMemcpy(m, device_ptr, sizeof(Model), cudaMemcpyDeviceToHost);

    // free everything
    cudaFree(m->indexes_);
    cudaFree(m->vectors_);
    free(m);
    cudaFree(device_ptr);

    if (m->diffuse_)
        TGAImage::cudaDeepFree(m->diffuse_);
    if (m->normal_map_)
        TGAImage::cudaDeepFree(m->normal_map_);
    if (m->specular_)
        TGAImage::cudaDeepFree(m->specular_);
}

__host__ __device__ TGAImage *Model::texture_of_type(TextureType type)
{
    switch (type)
    {
    case TextureType::DIFFUSE:
        return diffuse_;
    case TextureType::NORMAL_MAP:
        return normal_map_;
    case TextureType::SPECULAR:
        return specular_;
    }
    return diffuse_;
}

void Model::load_texture(const char *filename, TextureType type)
{
    TGAImage *texture = texture_of_type(type);

    texture->read_tga_file(filename);
    texture->flip_vertically(); // so the origin is left bottom corner
}

__host__ __device__ TGAColor Model::sample_texture(Vec2f uv, TextureType type)
{
    TGAImage *texture = texture_of_type(type);
    if (texture == nullptr)
    {
        return TGAColor(0, 0, 0, 0);
    }

    return texture->get(
        std::round(uv.x * (float)texture->get_width()),
        std::round(uv.y * (float)texture->get_height()));
}
