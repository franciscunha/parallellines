// This file is forked from an initial version by Dmitry V. Sokolov (https://github.com/ssloy) that was
// provided as a starting point for the tinyrenderer course (https://github.com/ssloy/tinyrenderer)

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include "../include/model.cuh"

Model::Model(const char *filename) : normal_map_(), specular_()
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
    std::vector<Vec2f> uvs;
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
            Vec2f vt;
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
}

Model::~Model()
{
    delete[] indexes_;
    delete[] vectors_;
}

__host__ __device__ TGAImage *Model::texture_of_type(TextureType type)
{
    switch (type)
    {
    case TextureType::DIFFUSE:
        return &diffuse_;
    case TextureType::NORMAL_MAP:
        return &normal_map_;
    case TextureType::SPECULAR:
        return &specular_;
    }
    return nullptr;
}

void Model::load_texture(const char *filename, TextureType type)
{
    TGAImage *texture = texture_of_type(type);
    if (texture == nullptr)
    {
        std::cerr << "texture type doesn't exist" << std::endl;
        return;
    }

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
