// layer.h		V.Rao, H. Rao
// header file for the layer class heirarchy and
// the network class

#define MAX_LAYERS	5
#define MAX_VECTORS	100
#include <cstdio>
#include <cstdlib>

class network;

class layer
{

protected:

	int num_inputs;
	int num_outputs;
	float *outputs;	// pointer to array of outputs
	float *inputs; 	// pointer to array of inputs, which
					// are outputs of some other layer
//friend class Singleton< Tableau >;
	friend class network;

public:

	virtual void calc_out()=0;
};



class input_layer: public layer
{

private:


public:

	input_layer(int, int);
	~input_layer();
	virtual void calc_out();

};

class middle_layer;

class output_layer:	public layer
{
protected:

	float * weights;
	float * output_errors; // array of errors at output
	float * back_errors; // array of errors back-propagated
	float * expected_values;	// to inputs
    friend class network;

public:


	output_layer(int, int);
	~output_layer();
	virtual void calc_out();
	void calc_error(float &);
	void randomize_weights();
	void update_weights(const float);
	void list_weights();
	void write_weights(int, FILE *);
	void read_weights(int, FILE *);
	void list_errors();
	void list_outputs();
};




class middle_layer:	public output_layer
{

private:

public:
    middle_layer(int, int);
    ~middle_layer();
	void calc_error();
};


class network

{

private:

layer *layer_ptr[MAX_LAYERS];
    int number_of_layers;
    int layer_size[MAX_LAYERS];
    float *buffer;
    //fpos_
    long position;
    unsigned training;

public:
    network();
    ~network();
	void set_training(const unsigned &);
	unsigned get_training_value();
	void get_layer_info();
	void set_up_network();
	void randomize_weights();
	void update_weights(const float);
	void write_weights(FILE *);
	void read_weights(FILE *);
	void list_weights();
	void write_outputs(FILE *);
	void list_outputs();
	void list_errors();
	void forward_prop();
	void backward_prop(float &);
	int fill_IObuffer(FILE *);
	void set_up_pattern(int);

};
