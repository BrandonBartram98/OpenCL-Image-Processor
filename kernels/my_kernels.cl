//Intense Kernel
kernel void intenseHist(global unsigned char* A, global int* B) {
	int id = get_global_id(0); //Get current global id
	int pixValue = (int)A[id]; //Pixel value

	atomic_inc(&B[pixValue]); //Atomic function
}

//Cumulative Kernel
kernel void cumulativeHist(global int* A, global int* B) {
	
	int id = get_global_id(0); //Variable = global id
	int pixSize = get_global_size(0); //No. pixels
	
	global int* C; //Cache buffer for A + B swap

	//For each item add largest val
	for (int stride = 1; stride < pixSize; stride *= 2) {
		//If id is greater or = to stride, add previous
		if (id >= stride) {
			B[id] = A[id] + A[id - stride];
		}
		else {
			B[id] = A[id];
		}

		barrier(CLK_GLOBAL_MEM_FENCE); //Syncronise steps
		//Swap data in buffer
		C = A;
		A = B;
		B = C;
	}
}

//Equalised Kernel
kernel void equalisedHist(global int* A, global int* B) {
	
	int id = get_global_id(0); //Get current global id
	int pixNumb = get_global_size(0); //pixNumb = number of pixels

	float currentPx = (float)A[id]; //Current pixel
	float lastPx = (float)A[pixNumb - 1]; //Last pixel

	B[id] = (int)(currentPx * 255 / lastPx); //Equalise pixel, scale 0 - 255

}

//Back Projection Kernel
kernel void backProj(global int* A, global unsigned char* B, global unsigned char* C) {
	//Compare to intensity hist, set to output 
	int id = get_global_id(0); //Get current global id
	int value = (int)B[id]; //Input value

	C[id] = (unsigned char)A[value]; //Equalised array value for input
}
