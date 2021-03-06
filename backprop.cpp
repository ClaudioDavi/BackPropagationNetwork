// backprop.cpp		V. Rao, H. Rao
#include "layer.h"
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <cmath>
#define TRAINING_FILE	"training.dat"
#define WEIGHTS_FILE "weights.dat"
#define OUTPUT_FILE	"output.dat"
#define TEST_FILE	"test.dat"
using namespace std;
int main()
{

float error_tolerance=0.1;
float total_error=0.0;
float avg_error_per_cycle=0.0;
float error_last_cycle=0.0;
float avgerr_per_pattern=0.0; // for the latest cycle
float error_last_pattern=0.0;
float learning_parameter=0.02;
unsigned temp, startup;
long int vectors_in_buffer;
long int max_cycles;
long int patterns_per_cycle=0;

long int total_cycles, total_patterns;
int i;


// create a network object
network backp;


FILE * training_file_ptr, * weights_file_ptr, * output_file_ptr;
FILE * test_file_ptr, * data_file_ptr;

// open output file for writing
if ((output_file_ptr=fopen(OUTPUT_FILE,"w"))==NULL)
		{
		cout << "problem opening output file\n";
		exit(1);
		}

// enter the training mode : 1=training on     0=training off
cout << "---------------------------------------------------\n";
cout << " C++ Neural Networks and Fuzzy Logic \n";
cout << " 	Backpropagation simulator \n";
cout << "		version 1 \n";
cout << "---------------------------------------------------\n";
cout << "Please enter 1 for TRAINING on, or 0 for off: \n\n";
cout << "Use training to change weights according to your\n";
cout << "expected outputs. Your training.dat file should contain\n";
cout << "a set of inputs and expected outputs. The number of\n";
cout << "inputs determines the size of the first (input) layer\n";
cout << "while the number of outputs determines the size of the\n";
cout << "last (output) layer :\n\n";

cin >> temp;
backp.set_training(temp);

if (backp.get_training_value() == 1)
	{
	cout << "--> Training mode is *ON*. weights will be saved\n";
	cout << "in the file weights.dat at the end of the\n";
	cout << "current set of input (training) data\n";
	}
else
	{
	cout << "--> Training mode is *OFF*. weights will be loaded\n";
	cout << "from the file weights.dat and the current\n";
	cout << "(test) data set will be used. For the test\n";
	cout << "data set, the test.dat file should contain\n";
	cout << "only inputs, and no expected outputs.\n";
	}

if (backp.get_training_value()==1)
	{
	// -----------------------------------------
	// 	Read in values for the error_tolerance,
	//	and the learning_parameter
	// -----------------------------------------
	cout << " Please enter in the error_tolerance\n";
	cout << " --- between 0.001 to 100.0, try 0.1 to start --\n";
	cout << "\n";
	cout << "and the learning_parameter, beta\n";
	cout << " --- between 0.01 to 1.0, try 0.5 to start -- \n\n";
	cout << " separate entries by a space\n";
	cout << " example: 0.1 0.5 sets defaults mentioned :\n\n";

	cin >> error_tolerance >> learning_parameter;
	//-------------------------------------------
	// open training file for reading
	//-------------------------------------------
	if ((training_file_ptr=fopen(TRAINING_FILE,"r"))==NULL)
		{
		cout << "problem opening training file\n";
		exit(1);
		}
	data_file_ptr=training_file_ptr; // training on

	// Read in the maximum number of cycles
	// each pass through the input data file is a cycle
	cout << "Please enter the maximum cycles for the simulation\n";
	cout << "A cycle is one pass through the data set.\n";
	cout << "Try a value of 10 to start with\n";

	cin >> max_cycles;

	}
else
	{
	if ((test_file_ptr=fopen(TEST_FILE,"r"))==NULL)
		{
		cout << "problem opening test file\n";
		exit(1);
		}

	data_file_ptr=test_file_ptr; // training off
	}




// the main loop
//
// training: continue looping until the total error is less than
// 		the tolerance specified, or the maximum number of
// 		cycles is exceeded; use both the forward signal propagation
// 		and the backward error propagation phases. If the error
//		tolerance criteria is satisfied, save the weights in a file.
// no training: just proceed through the input data set once in the
// 		forward signal propagation phase only. Read the starting
//		weights from a file.
// in both cases report the outputs on the screen


// intialize counters
total_cycles=0; // a cycle is once through all the input data
total_patterns=0; // a pattern is one entry in the input data



// get layer information
backp.get_layer_info();

// set up the network connections
backp.set_up_network();

// initialize the weights
if (backp.get_training_value()==1)
	{
	// randomize weights for all layers; there is no
	// weight matrix associated with the input layer
	// weight file will be written after processing
	// so open for writing
	if ((weights_file_ptr=fopen(WEIGHTS_FILE,"w"))
			==NULL)
		{
		cout << "problem opening weights file\n";
		exit(1);
		}
	backp.randomize_weights();
	}
else
	{
	// read in the weight matrix defined by a
	// prior run of the backpropagation simulator
	// with training on
	if ((weights_file_ptr=fopen(WEIGHTS_FILE,"r"))
			==NULL)
		{
		cout << "problem opening weights file\n";
		exit(1);
		}
	backp.read_weights(weights_file_ptr);
	}




// main loop
// if training is on, keep going through the input data
// 		until the error is acceptable or the maximum number of cycles
// 		is exceeded.
// if training is off, go through the input data once. report outputs
// with inputs to file output.dat

startup=1;
vectors_in_buffer = MAX_VECTORS; // startup condition
total_error = 0;


while (			((backp.get_training_value()==1)
			&& (avgerr_per_pattern
					> error_tolerance)
			&& (total_cycles < max_cycles)
			&& (vectors_in_buffer !=0))
			|| ((backp.get_training_value()==0)
			&& (total_cycles < 1))
			|| ((backp.get_training_value()==1)
			&& (startup==1))
			)
{
startup=0;
error_last_cycle=0; // reset for each cycle
patterns_per_cycle=0;
// process all the vectors in the datafile
// going through one buffer at a time
// pattern by pattern



while ((vectors_in_buffer==MAX_VECTORS))
	{

	vectors_in_buffer=
		backp.fill_IObuffer(data_file_ptr); // fill buffer
		if (vectors_in_buffer < 0)
			{
			cout << "error in reading in vectors, aborting\n";
			cout << "check that there are on extra linefeeds\n";
			cout << "in your data file, and that the number\n";
			cout << "of layers and size of layers match the\n";
			cout << "the parameters provided.\n";
			exit(1);
			}

		// process vectors
		for (i=0; i<vectors_in_buffer; i++)
			{
			// get next pattern
			backp.set_up_pattern(i);

			total_patterns++;
			patterns_per_cycle++;
			// forward propagate

			backp.forward_prop();

			if (backp.get_training_value()==0)
				backp.write_outputs(output_file_ptr);

			// back_propagate, if appropriate
			if (backp.get_training_value()==1)
				{

				backp.backward_prop(error_last_pattern);
				error_last_cycle +=
					error_last_pattern*error_last_pattern;
				backp.update_weights(learning_parameter);
				// backp.list_weights(); // can
				// see change in weights by
				// using list_weights before and
				// after back_propagation
				}

			}

	error_last_pattern = 0;
    	}

avgerr_per_pattern=((float)sqrt((double)error_last_cycle/patterns_per_cycle));
total_error += error_last_cycle;
total_cycles++;

// most character displays are 25 lines
// user will see a corner display of the cycle count
// as it changes

cout << "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n";
cout << total_cycles << "\t" << avgerr_per_pattern << "\n";

fseek(data_file_ptr, 0L, SEEK_SET); // reset the file pointer
				// to the beginning of
				// the file
vectors_in_buffer = MAX_VECTORS; // reset

} // end main loop

cout << "\n\n\n\n\n\n\n\n\n\n\n";
cout << "---------------------------------------------------\n";
cout << " 	done:	results in file output.dat\n";
cout << "		training: last vector only\n";
cout << "		not training: full cycle\n\n";
if (backp.get_training_value()==1)
	{
	backp.write_weights(weights_file_ptr);
	backp.write_outputs(output_file_ptr);
	avg_error_per_cycle= (float)sqrt((double)total_error/total_cycles);
	error_last_cycle=(float)sqrt((double)error_last_cycle);

cout << "		weights saved in file weights.dat\n";
cout << "\n";
cout << "---->average error per cycle = " << avg_error_per_cycle << " <---\n";
cout << "---->error last cycle = " << error_last_cycle << " <---\n";
cout << "->error last cycle per pattern= " << avgerr_per_pattern << " <---\n";

	}

cout << "------------>total cycles = " << total_cycles << " <---\n";
cout << "------------>total patterns = " << total_patterns << " <---\n";
cout << "---------------------------------------------------\n";
// close all files
fclose(data_file_ptr);
fclose(weights_file_ptr);
fclose(output_file_ptr);

}
