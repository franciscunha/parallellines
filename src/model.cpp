// This file is written by Dmitry V. Sokolov (https://github.com/ssloy) and provided
// as a starting point for the tinyrenderer course (https://github.com/ssloy/tinyrenderer)

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include "model.hpp"

Model::Model(const char *filename) : verts_(), uvs_(), faces_(), faces_uvs_(), texture_() {
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
        } else if (!line.compare(0, 2, "f ")) {
            std::vector<int> v_indexes;
            std::vector<int> t_indexes;
            int itrash, v_idx, t_idx;
            iss >> trash;
            while (iss >> v_idx >> trash >> t_idx >> trash >> itrash) {
                // in wavefront obj all indices start at 1, not zero
                v_idx--;
                t_idx--; 

                v_indexes.push_back(v_idx);
                t_indexes.push_back(t_idx);
            }
            faces_.push_back(v_indexes);
            faces_uvs_.push_back(t_indexes);
        }
    }
    std::cout << "# v# " << verts_.size() << " f# "  << faces_.size() << " t# " << uvs_.size() << std::endl;
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

Vec3f Model::vert(int i) {
    return verts_[i];
}

Vec2f Model::uv(int i) {
    return uvs_[i];
}

void Model::load_texture(const char *filename) {
	texture_.read_tga_file(filename);
	texture_.flip_vertically(); // so the origin is left bottom corner
}

TGAColor Model::sample_texture(Vec2f uv) {
    return texture_.get(
        std::round(uv.x * (float)texture_.get_width()), 
        std::round(uv.y * (float)texture_.get_height())
    );
}