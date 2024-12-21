#include "../include/line_renderer.cuh"
#include "../include/thread_count.cuh"

namespace line_renderer
{
	namespace // private functions
	{
		__device__ Vec2i world_to_screen_coords(Vec3f world_coords, int width, int height)
		{
			return Vec2i(
				roundf((world_coords.x + 1.0f) * (float)width / 2.0f),
				roundf((world_coords.y + 1.0f) * (float)height / 2.0f));
		}

		__host__ __device__ void swap(int *a, int *b)
		{
			int c = *a;
			*a = *b;
			*b = c;
		}

		__device__ void d_draw(Vec2i p0, Vec2i p1, TGAImage *output, TGAColor color)
		{
			int x0 = p0.x, y0 = p0.y;
			int x1 = p1.x, y1 = p1.y;

			bool steep = false;
			if (abs(x0 - x1) < abs(y0 - y1))
			{
				steep = true;
				swap(&x0, &y0);
				swap(&x1, &y1);
			}

			if (x1 < x0)
			{
				swap(&x0, &x1);
				swap(&y0, &y1);
			}

			float slope = (y0 - y1) / (float)(x0 - x1);
			float y = y0;

			for (int x = x0; x <= x1; x++)
			{
				if (!steep)
				{
					output->set(x, roundf(y), color);
				}
				else
				{
					output->set(roundf(y), x, color);
				}
				y = y + slope;
			}
		}

		__global__ void wireframe_kernel(Model *model, TGAImage *output, TGAColor color, int nfaces, int width, int height)
		{
			int idx = blockIdx.x * blockDim.x + threadIdx.x;
			if (idx >= nfaces)
				return;

			int *face_vertices = model->face(idx); 

			for (int j = 0; j < 3; j++)
			{
				Vec2i vertex0 = world_to_screen_coords(model->vert(face_vertices[j]), width, height);
				Vec2i vertex1 = world_to_screen_coords(model->vert(face_vertices[(j + 1) % 3]), width, height);
				d_draw(vertex0, vertex1, output, color);
			}

		}
	}

	void draw(Vec2i p0, Vec2i p1, TGAImage &output, TGAColor color)
	{
		int x0 = p0.x, y0 = p0.y;
		int x1 = p1.x, y1 = p1.y;

		bool steep = false;
		if (std::abs(x0 - x1) < std::abs(y0 - y1))
		{
			steep = true;
			std::swap(x0, y0);
			std::swap(x1, y1);
		}

		if (x1 < x0)
		{
			std::swap(x0, x1);
			std::swap(y0, y1);
		}

		float slope = (y0 - y1) / (float)(x0 - x1);
		float y = y0;

		for (int x = x0; x <= x1; x++)
		{
			if (!steep)
			{
				output.set(x, std::round(y), color);
			}
			else
			{
				output.set(std::round(y), x, color);
			}
			y = y + slope;
		}
	}

	void wireframe(Model *model, TGAImage *output, TGAColor color)
	{		
		int w = output->get_width();
		int h = output->get_height();

		// create device pointers for parameters
		TGAImage *d_output_image = output->cudaDeepCopyToDevice();
		Model *d_model = model->cudaDeepCopyToDevice();
		
		// call kernel
		size_t num_blocks, threads_per_block;
		calculate_kernel_size(model->nfaces(), &num_blocks, &threads_per_block);
		wireframe_kernel<<<num_blocks, threads_per_block>>>(d_model, d_output_image, color, model->nfaces(), w, h);

		// make sure faces are rendered
		cudaDeviceSynchronize();

		// copy result back to output image
		output->cudaDeepCopyFromDevice(*d_output_image);

		// free device memory
		TGAImage::cudaDeepFree(d_output_image);
		Model::cudaDeepFree(d_model);
	}
}