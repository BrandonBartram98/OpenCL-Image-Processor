#include <iostream>
#include <vector>
#include <string>
#include "CImg.h"
#include "Utils.h"

using namespace cimg_library;

/*

Parallel Programming - BAR15623614 Brandon Bartram

Application tested on GTX 1070 graphics card

Application features working histogram equalisation tool using the atomic functions. The application works with the provided assignment images.
The memory transfer and kernel execution times are provided after each kernel has ran. All basic assignment requirements are completed.
The application also works with coloured images. This can be tested by changing the imageFileName variable to "colourTest.ppm".

The intensity histogram is calculated from the input image using the atomic function, the cumulative histogram is then calculated.
The cumulative histogram is then scaled and normalised. The equaliser kernel runs through the pixels scale from 0 to 255 to equalise. 
The back projection kernel is used to map the original intensities onto the output by comparing it to the current state.
The output is equalised and displayed, the console provides information on memory transfer and kernel execution time for each kernel.

*/

int main(int argc, char** argv) {

	while (main) {
		try {
			cl::Context context = GetContext(0, 0);
			cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);
			cl::Program::Sources sources;

			AddSources(sources, "kernels/my_kernels.cl");

			cl::Program program(context, sources);
			cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
			try {
				program.build();
			}
			//Error management
			catch (const cl::Error & err) {
				std::cout << "build status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << endl;

				std::cout << "build option:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << endl;

				std::cout << "build log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << endl;

				throw err;
			}

			//VARIABLES
			int bin = 256; //0-255 pixel values
			int i = 0;

			//Change imageFileName to colourTest to test with coloured image
			string imageFileName = "test.pgm"; //Import image

			CImg<unsigned char> imageInput(imageFileName.c_str()); //Input image object
			vector<int> intenseHist(bin, 0); //Intensity histogram vector
			vector<int> cumulativeHist(bin, 0);	//Cumulative histogram vector
			vector<int> equalisedHist(bin, 0);	//Equalised histogram vector

			//BUFFERS
			cl::Buffer inputBuff(context, CL_MEM_READ_ONLY, imageInput.size()); //Input buffer
			cl::Buffer outputBuff(context, CL_MEM_READ_WRITE, imageInput.size()); //Output buffer
			cl::Buffer histBuff(context, CL_MEM_READ_WRITE, bin * sizeof(int)); //Histogram buffer
			cl::Buffer cumulativeHistBuff(context, CL_MEM_READ_WRITE, bin * sizeof(int)); //Cumulative histogram buffer
			cl::Buffer equalisedHistBuff(context, CL_MEM_READ_WRITE, bin * sizeof(int)); //Equalised histogram buffer

			vector<unsigned char> outputBuffer(imageInput.size()); //Output image buffer

			cl::Event event;

			//KERNELS
			//Intense Kernel
			cl::Kernel kerIntenseHist = cl::Kernel(program, "intenseHist");
			//Give kernel arguments
			kerIntenseHist.setArg(0, inputBuff);
			kerIntenseHist.setArg(1, histBuff);

			//Cumulative Kernel
			cl::Kernel kerCumulativeHist = cl::Kernel(program, "cumulativeHist");
			kerCumulativeHist.setArg(0, histBuff);
			kerCumulativeHist.setArg(1, cumulativeHistBuff);

			//Equalise Kernel
			cl::Kernel kerEqualisedHist = cl::Kernel(program, "equalisedHist");
			kerEqualisedHist.setArg(0, cumulativeHistBuff);
			kerEqualisedHist.setArg(1, equalisedHistBuff);

			//Back Projection Kernel
			cl::Kernel kerBackProj = cl::Kernel(program, "backProj");
			kerBackProj.setArg(0, equalisedHistBuff);
			kerBackProj.setArg(1, inputBuff);
			kerBackProj.setArg(2, outputBuff);

			CImgDisplay showInput(imageInput, "Input"); //Display input image

			std::cout << ListPlatformsDevices() << std::endl; //Display available platforms/devices

			//INTENSITY HISTOGRAM
			cout << "Kernel 1 Start" << endl;
			//Add variables to buffers
			queue.enqueueWriteBuffer(inputBuff, CL_TRUE, 0, imageInput.size(), &imageInput.data()[0]);
			queue.enqueueWriteBuffer(histBuff, CL_TRUE, 0, bin * sizeof(int), &intenseHist[0]);
			queue.enqueueNDRangeKernel(kerIntenseHist, cl::NullRange, cl::NDRange(imageInput.size()), cl::NullRange, NULL, &event); //Run kernel with event
			//Read and store buffer output into vector
			queue.enqueueReadBuffer(histBuff, CL_TRUE, 0, bin * sizeof(int), &intenseHist[0]);
			//Display kernel results
			cout << "Kernel 1:" << endl << intenseHist << endl << GetFullProfilingInfo(event, ProfilingResolution::PROF_NS);


			//CUMULATIVE HISTOGRAM
			cout << "\nKernel 2 Start" << endl;
			//Add variables to buffers
			queue.enqueueWriteBuffer(histBuff, CL_TRUE, 0, bin * sizeof(int), &intenseHist[0]);
			queue.enqueueWriteBuffer(cumulativeHistBuff, CL_TRUE, 0, bin * sizeof(int), &cumulativeHist[0]);
			queue.enqueueNDRangeKernel(kerCumulativeHist, cl::NullRange, cl::NDRange(bin), cl::NullRange, NULL, &event); //Run kernel with event
			//Read and store buffer output into vector
			queue.enqueueReadBuffer(cumulativeHistBuff, CL_TRUE, 0, bin * sizeof(int), &cumulativeHist[0]);
			//Fix for black spot errors
			for (i = 1; i < 256; i++)
			{
				if (cumulativeHist[i] < cumulativeHist[i - 1])
				{
					cumulativeHist[i] = cumulativeHist[i - 1];
				}
			}
			//Display kernel results
			cout << "Kernel 2:" << endl << cumulativeHist << endl << GetFullProfilingInfo(event, ProfilingResolution::PROF_NS);


			//EQUALISE HISTOGRAM
			cout << "\nKernel 3 Start" << endl;
			//Add variables to buffers
			queue.enqueueWriteBuffer(cumulativeHistBuff, CL_TRUE, 0, bin * sizeof(int), &cumulativeHist[0]);
			queue.enqueueNDRangeKernel(kerEqualisedHist, cl::NullRange, cl::NDRange(bin), cl::NullRange, NULL, &event); //Run kernel with event
			//Read and store buffer output into vector
			queue.enqueueReadBuffer(equalisedHistBuff, CL_TRUE, 0, bin * sizeof(int), &equalisedHist[0]);
			//Display kernel results
			cout << "Kernel 3:" << endl << equalisedHist << endl << GetFullProfilingInfo(event, ProfilingResolution::PROF_NS);


			//BACK PROJECTION
			cout << "Kernel 4 Start" << endl;
			//Add variables to buffers
			queue.enqueueWriteBuffer(equalisedHistBuff, CL_TRUE, 0, bin * sizeof(int), &equalisedHist[0]);
			queue.enqueueNDRangeKernel(kerBackProj, cl::NullRange, cl::NDRange(imageInput.size()), cl::NullRange, NULL, &event);; //Run kernel with event
			//Read and store buffer output into vector
			queue.enqueueReadBuffer(outputBuff, CL_TRUE, 0, outputBuffer.size(), &outputBuffer.data()[0]);
			//Display kernel results
			cout << "Kernel 4:" << endl <<  GetFullProfilingInfo(event, ProfilingResolution::PROF_NS);


			//SHOW IMAGES
			CImg<unsigned char> output_image(outputBuffer.data(), imageInput.width(), imageInput.height(), imageInput.depth(), imageInput.spectrum());
			CImgDisplay showOutput(output_image, "Equalised Output");

			//Wait until I/O closed 
			while (!showInput.is_closed() && !showInput.is_keyESC() && !showOutput.is_closed() && !showInput.is_keyESC()) {
				showInput.wait(1);
				showOutput.wait(1);
			}
		}

		catch (const cl::Error & err) {
			std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << endl;
		}
		catch (CImgException & err) {
			std::cerr << "ERROR: " << err.what() << endl;
		}
	}
	return 0;
}