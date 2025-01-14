#include <cmath>
#include <utility>
#include <iostream>
#include <algorithm>

#include "../include/tgaimage.cuh"
#include "../include/model.cuh"
#include "../include/line_renderer.cuh"

#include "../include/renderer.cuh"
#include "../include/shaders.cuh"

void renderer_demo(
	int output_width,
	int output_height,
	char *model_filepath,
	char *diffuse_filepath,
	char *specular_filepath,
	char *normal_map_filepath)
{
	TGAImage output(output_width, output_height, TGAImage::RGB);

	Model model(model_filepath);
	model.load_texture(diffuse_filepath, TextureType::DIFFUSE);
	model.load_texture(specular_filepath, TextureType::SPECULAR);
	model.load_texture(normal_map_filepath, TextureType::NORMAL_MAP);

	renderer::render<PhongShader, PhongShaderData>(output, model);

	output.flip_vertically(); // so the origin is left bottom corner
	output.write_tga_file("images/out.tga");
}

void lines_demo(int output_width, int output_height, char *model_filepath)
{
	TGAImage output(output_width, output_height, TGAImage::RGB);
	Model model(model_filepath);

	line_renderer::wireframe(&model, &output, TGAColor(255, 255, 255, 255));

	output.flip_vertically(); // so the origin is left bottom corner
	output.write_tga_file("images/out.tga");
}

void append_c_str(const char *str1, const char *str2, char *out)
{
	strcpy(out, str1);
	strcat(out, str2);
}

int main(int argc, char **argv)
{
	if (argc != 5)
	{
		std::cout << "USAGE: parallelllines.exe [render|wireframe] WIDTH HEIGHT DIR_PATH" << std::endl;
		return 1;
	}

	const int width = strtol(argv[2], nullptr, 10);
	const int height = strtol(argv[3], nullptr, 10);

	char model_filepath[100];
	char diffuse_filepath[100];
	char specular_filepath[100];
	char normal_map_filepath[100];
	
	append_c_str(argv[4], "/model.obj", model_filepath);
	append_c_str(argv[4], "/diffuse.tga", diffuse_filepath);
	append_c_str(argv[4], "/specular.tga", specular_filepath);
	append_c_str(argv[4], "/normal_map.tga", normal_map_filepath);

	if (argv[1][0] == 'r')
	{
		// render
		renderer_demo(width, height, model_filepath, diffuse_filepath, specular_filepath, normal_map_filepath);
	}
	else if (argv[1][0] == 'w')
	{
		// wireframe
		lines_demo(width, height, model_filepath);
	}
	else 
	{
		// invalid option
		std::cout << "USAGE: parallelllines.exe [render|wireframe] WIDTH HEIGHT DIR_PATH" << std::endl;
		return 1;
	}

	return 0;
}
