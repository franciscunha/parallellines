// This file is forked from an initial version by Dmitry V. Sokolov (https://github.com/ssloy) that was
// provided as a starting point for the tinyrenderer course (https://github.com/ssloy/tinyrenderer)

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include "model.hpp"

Model::Model(const char *filename) : verts_(), uvs_(), faces_(), faces_uvs_(), diffuse_(), normal_map_(), specular_()  {
    std::ifstream in;
    in.open (filename, std::ifstream::in);
    if (in.fail()) {
        std::cerr << "failed to open " << filename << std::endl;
        return;
    }
    
    std::string line;
    while (!in.eof()) {
        std::getline(in, line);
        std::istringstream iss(line.c_str());
        
        char trash;
        if (!line.compare(0, 2, "v ")) {
            iss >> trash;
            Vec3f v;
            for (int i=0;i<3;i++) iss >> v.raw[i];
            verts_.push_back(v);
        } else if (!line.compare(0, 3, "vt ")) {
            iss >> trash >> trash;
            Vec2f vt;
            for (int i=0;i<2;i++) iss >> vt.raw[i];
            uvs_.push_back(vt);
        } else if (!line.compare(0, 3, "vn ")) {
            iss >> trash >> trash;
            Vec3f vn;
            for (int i=0;i<3;i++) iss >> vn.raw[i];
            normals_.push_back(vn);
        } else if (!line.compare(0, 2, "f ")) {
            std::vector<int> v_indexes;
            std::vector<int> t_indexes;
            std::vector<int> n_indexes;
            int v_idx, t_idx, n_idx;
            iss >> trash;
            while (iss >> v_idx >> trash >> t_idx >> trash >> n_idx) {
                // in wavefront obj all indices start at 1, not zero
                v_idx--;
                t_idx--; 
                n_idx--;

                v_indexes.push_back(v_idx);
                t_indexes.push_back(t_idx);
                n_indexes.push_back(n_idx);
            }
            faces_.push_back(v_indexes);
            faces_uvs_.push_back(t_indexes);
            faces_normals_.push_back(n_indexes);
        }
    }
}

Model::~Model() {
}

int Model::nverts() {
    return (int)verts_.size();
}

int Model::nfaces() {
    return (int)faces_.size();
}

std::vector<int> Model::face(int idx) {
    return faces_[idx];
}

std::vector<int> Model::face_uvs(int idx) {
    return faces_uvs_[idx];
}

std::vector<int> Model::face_normals(int idx) {
    return faces_normals_[idx];
}

Vec3f Model::vert(int i) {
    return verts_[i];
}

Vec2f Model::uv(int i) {
    return uvs_[i];
}

Vec3f Model::normal(int i) {
    return normals_[i];
}

TGAImage *Model::texture_of_type(TextureType type) {
    switch (type) {
        case TextureType::DIFFUSE:
            return &diffuse_;
        case TextureType::NORMAL_MAP:
            return &normal_map_;
        case TextureType::SPECULAR:
            return &specular_;
    }
    return NULL;
}

void Model::load_texture(const char *filename, TextureType type) {
    TGAImage *texture = texture_of_type(type);
    if (texture == NULL) {
        std::cerr << "texture type doesn't exist" << std::endl;
        return;
    }

    texture->read_tga_file(filename);
	texture->flip_vertically(); // so the origin is left bottom corner
}

TGAColor Model::sample_texture(Vec2f uv, TextureType type) {
    TGAImage *texture = texture_of_type(type);
    if (texture == NULL) {
        std::cerr << "texture type doesn't exist" << std::endl;
        return TGAColor(0, 0, 0, 0);
    }

    return texture->get(
        std::round(uv.x * (float)texture->get_width()), 
        std::round(uv.y * (float)texture->get_height())
    );
}
