#include<bits/stdc++.h>
#include<sys/time.h>
#include<omp.h>
using namespace std;
#define datasize 2500
#define epochs 20
#define batchsize 500
//good result at datasize 10000 epochs 20 batchsize 500 learning rate 0.1 layers 784 100 10 func no 1 i.e. sigmoid

vector<vector<double>> parallel_transpose(vector<vector<double>> &A)
{
	int m=A.size();
	int n=A[0].size();
	// cout<<"\nA:\t"<<m<<" "<<n;
	vector<vector<double>> trans(n,vector<double>(m));
	# pragma omp parallel for collapse(2)
	for(int i=0;i<m;i++)
		for(int j=0;j<n;j++)
			trans[j][i] = A[i][j];
	return trans;
}

vector<vector<double>> parallel_dot(vector<vector<double>> &A,vector<vector<double>> &B)
{
	int m=A.size();
	int n=A[0].size();
	int p=B.size();
	int q=B[0].size();
	if(n!=p){cout<<"\nNot possible to multiply!\n";exit(0);}
	vector<vector<double>> C(m,vector<double>(q));
	# pragma omp parallel for collapse(2)
	for(int i=0;i<m;i++)
		for(int j=0;j<q;j++)
			for(int k=0;k<n;k++)
				C[i][j] += A[i][k] * B[k][j];
	// for(int i=0;i<m;i++)
	// 	for(int j=0;j<q;j++)
	// 		cout<<C[i][j]<<" ";
	// exit(0);
	return C;
}

void parallel_broadcast_multiply_value(vector<vector<double>> &A,double a,double b=1.0)
{
	int m=A.size();
	int n=A[0].size();
	// cout<<a<<endl;
	// cout<<b<<endl;

	# pragma omp parallel for collapse(2)
	for(int i=0;i<m;i++)
		for(int j=0;j<n;j++)
			A[i][j] = a*b*A[i][j];
}

void parallel_broadcast_add_matrix_vector(vector<vector<double>> &A,vector<double> &B)
{
	int m=A.size();
	int n=A[0].size();
	int p=B.size();
	if(m!=p){cout<<"\nNot possible to add!\n";exit(0);}

	# pragma omp parallel for collapse(2)
	for(int i=0;i<m;i++)
		for(int j=0;j<n;j++)
			A[i][j] += B[i];
}

void parallel_broadcast_add_matrixs(vector<vector<double>> &A,vector<vector<double>> &B)
{
	int m=A.size();
	int n=A[0].size();
	int p=B.size();
	int q=B[0].size();
	if(m!=p&&n!=q){cout<<"\nNot possible to difference!\n";exit(0);}
	# pragma omp parallel for collapse(2)
	for(int i=0;i<m;i++)
		for(int j=0;j<n;j++)
				A[i][j] += B[i][j];
}

void parallel_broadcast_multiply_matrixs(vector<vector<double>> &A,vector<vector<double>> &B)
{
	int m=A.size();
	int n=A[0].size();
	int p=B.size();
	int q=B[0].size();
	if(m!=p&&n!=q){cout<<"\nNot possible to difference!\n";exit(0);}
	# pragma omp parallel for collapse(2)
	for(int i=0;i<m;i++)
		for(int j=0;j<n;j++)
				A[i][j] *= B[i][j];
}

void parallel_broadcast_add_vectors(vector<double> &A,vector<double> &B)
{
	int m=A.size();
	int p=B.size();
	if(m!=p){cout<<"\nNot possible to difference!\n";exit(0);}
	# pragma omp parallel for
	for(int i=0;i<m;i++)
		A[i] += B[i];
}

double sigmoid(double value)
{
	return 1.0/(1.0+exp(-1.0*value));
}


void parallel_activation(vector<vector<double>> &A,int func_no)
{
	int m=A.size();
	int n=A[0].size();
	switch(func_no)
	{
		case 1:
		# pragma omp parallel for collapse(2)
		for(int i=0;i<m;i++)
			for(int j=0;j<n;j++)
				A[i][j] = sigmoid(A[i][j]);
		break;
		case 2://relu
		# pragma omp parallel for collapse(2)
		for(int i=0;i<m;i++)
			for(int j=0;j<n;j++)
				if(A[i][j]<0.0)
					A[i][j] = 0.0;
		break;
	}
	
}

double sigmoid_prime(double value)
{
	return value*(1.0-value);
}

double relu_prime(double value)
{
	if(value>0.0)return 0.0;
	return 1.0;
}

void parallel_activation_prime(vector<vector<double>> &A,int func_no)
{
	int m=A.size();
	int n=A[0].size();
	switch(func_no)
	{
		case 1:
		# pragma omp parallel for collapse(2)
		for(int i=0;i<m;i++)
			for(int j=0;j<n;j++)
				A[i][j] = sigmoid_prime(A[i][j]);
		break;
		case 2:
		# pragma omp parallel for collapse(2)
		for(int i=0;i<m;i++)
			for(int j=0;j<n;j++)
				A[i][j] = relu_prime(A[i][j]);
		break;
	}
	
}

vector<double> parallel_mean_row_wise(vector<vector<double>> &A,double learning_rate)
{
	int m=A.size();
	int n=A[0].size();
	vector<double> mean(m);
	# pragma omp parallel for
	for(int i=0;i<m;i++)
	{	
		for(int j=0;j<n;j++)
			mean[i] += A[i][j];
		mean[i]=mean[i]*learning_rate/n;
	}
	return mean;

}

void parallel_softmax(vector<vector<double>> &A)
{
	int m=A.size();
	int n=A[0].size();
	vector<double> maxi(n,-INFINITY);
	vector<double> sum(n);
	# pragma omp parallel
	{
	# pragma omp for
	for(int i=0;i<n;i++)
		for(int j=0;j<m;j++)
			maxi[i] = max(maxi[i],A[j][i]);
	# pragma omp for
	for(int i=0;i<n;i++)
		for(int j=0;j<m;j++)
			{A[j][i] = exp(A[j][i]-maxi[i]);
			sum[i] += A[j][i];}
	# pragma omp for
	for(int i=0;i<n;i++)
		for(int j=0;j<m;j++)
			{A[j][i] /= sum[i];}
	}
}

vector<vector<double>> parallel_difference(vector<vector<double>> &A,vector<vector<double>> &B)
{
	int m=A.size();
	int n=A[0].size();
	int p=B.size();
	int q=B[0].size();
	if(m!=p&&n!=q){cout<<"\nNot possible to difference!\n";exit(0);}
	vector<vector<double>> C(m,vector<double>(n));
	# pragma omp parallel for collapse(2)
	for(int i=0;i<m;i++)
		for(int j=0;j<n;j++)
				C[i][j] = (A[i][j] - B[i][j]);
	return C;
}

class NeuralNetwork
{
	public:
	vector<int> layers;
	double learning_rate;
	int func_no;
	vector<vector<vector<double>>> weights;
	vector<vector<double>> bais;
	public:
	NeuralNetwork(vector<int> layers,double l,int no)
	{
		this->layers = layers;
		this->learning_rate = l;
		this->func_no = no;
		for(int i=0;i<layers.size()-1;i++)
		{
			vector<vector<double>> w(layers[i+1],vector<double>(layers[i]));
			vector<double> b(layers[i+1]);
			for (int j =0; j < layers[i+1]; ++j) {
	            for (int k =0; k < layers[i]; ++k) {
	                int sign = rand() % 2;
	                w[j][k] = (double)((rand() % 11)-5) / 500.0;
	                if (sign == 1) {
	                    w[j][k] = - w[j][k];
	                }
           		}
			}

		this->weights.push_back(w);
		this->bais.push_back(b);
		}
	}
	void feed_foreward(vector<vector<double>> &x,vector<vector<double>> &predict,vector<vector<vector<double>>> &inputs,vector<vector<vector<double>>> &outputs)
	{
		predict = parallel_transpose(x);
		inputs.push_back(predict);
		int i;
		for(i=0;i<this->weights.size()-1;i++)
		{
			// cout<<"\nweights:\t"<<weights[i].size()<<" "<<weights[i][0].size();
			predict = parallel_dot(this->weights[i],predict);
			// cout<<"\nBias:\t"<<bais[i].size();
			parallel_broadcast_add_matrix_vector(predict,this->bais[i]);
			parallel_activation(predict,this->func_no);

			outputs.push_back(predict);
			inputs.push_back(predict);
		}
		predict = parallel_dot(this->weights[i],predict);
		parallel_broadcast_add_matrix_vector(predict,this->bais[i]);

		// parallel_activation(predict,this->func_no);
		parallel_softmax(predict);
		// for(auto row:predict)
	 //   		for(auto col:row)
	 //   			cout<<col<<" ";
		// 		exit(0);
		
		outputs.push_back(predict);
		// cout<<"\npredict:\t"<<predict.size()<<" "<<predict[0].size()<<"\n";
		// for(int i=0;i<10;i++)cout<<predict[i][0]<<" ";
	}
	void train(vector<vector<double>> &x,vector<vector<double>> &y)
	{
		vector<vector<double>> predict,target,error,gradient,delta_weight,temp;
		vector<vector<vector<double>>> inputs,outputs,errors;
		feed_foreward(x,predict,inputs,outputs);
		
		target = parallel_transpose(y);
		error = parallel_difference(target,predict);
		errors.push_back(error);

		// cout<<"\npredict:\t"<<predict.size()<<" "<<predict[0].size()<<"\n";
		// cout<<"\ntarget:\t"<<target.size()<<" "<<target[0].size()<<"\n";
		// cout<<".";
		// cout.flush();

		for(int i=this->weights.size()-1;i>0;i--)
		{
			temp = parallel_transpose(this->weights[i]);
			error = parallel_dot(temp,error);
			errors.push_back(error);
		}

		reverse(errors.begin(),errors.end());
		int i = this->weights.size()-1;
		gradient = errors[i];
		temp = parallel_transpose(inputs[i]);
		delta_weight = parallel_dot(gradient,temp);
		parallel_broadcast_multiply_value(delta_weight,this->learning_rate,1.0/batchsize);
		parallel_broadcast_add_matrixs(this->weights[i],delta_weight);

		vector<double> mean_gradient = parallel_mean_row_wise(gradient,this->learning_rate);
		
		parallel_broadcast_add_vectors(this->bais[i],mean_gradient);
		// for(auto m:this->bais[i])cout<<m<<" ";
  //  		exit(0);
		i--;
		// cout<<mean_gradient.size()<<this->bais[i].size();

		for(;i>=0;i--)
		{
			parallel_activation_prime(outputs[i],this->func_no);
			// for(auto row:outputs[i])
   // 		for(auto col:row)
   // 			cout<<col<<" ";
			// exit(0);
			gradient = errors[i];
			parallel_broadcast_multiply_matrixs(gradient,outputs[i]);
			temp = parallel_transpose(inputs[i]);
			delta_weight = parallel_dot(gradient,temp);
			parallel_broadcast_multiply_value(delta_weight,this->learning_rate,(1.0/batchsize));
			parallel_broadcast_add_matrixs(this->weights[i],delta_weight);
			vector<double> mean_gradient = parallel_mean_row_wise(gradient,this->learning_rate);
			parallel_broadcast_add_vectors(this->bais[i],mean_gradient);
		}

	}
};

vector<int> argument_max(vector<vector<double>> &A)
{
	int m=A.size();
	int n=A[0].size();
	vector<int> result(n);
	# pragma omp parallel for
	for(int i=0;i<n;i++)
	{	double maxi=0;
		for(int j=0;j<m;j++)
			if(A[j][i]>maxi)
			{
				maxi=A[j][i];
				result[i]=j;
			}
			
	}
	return result;
}

int gen(int j)
{ 
    return rand()%j; 
}

void train_test_split(vector<vector<double>> &data,vector<vector<double>> &x_train,vector<vector<double>> &x_test,vector<vector<double>> &y_train,vector<vector<double>> &y_test)
{
    int i;
  	for(i=datasize-1;i>0;--i) {
    	swap(data[i],data[gen(i+1)]);
  	}

    vector<vector<double>> y(datasize,vector<double>(10));
	for(i=0;i<datasize;i++)
		y[i][data[i][0]]=1;

	for(i=0;i<datasize;i++)
		data[i].erase(data[i].begin());
	
	for(int i=0;i<floor(0.8*datasize);i++)
		{x_train.push_back(data[i]);
		y_train.push_back(y[i]);}

	for(int i=floor(0.8*datasize);i<datasize;i++)
		{x_test.push_back(data[i]);
		y_test.push_back(y[i]);}
} 

int main()
{
	struct timeval start, end;			//for time calculation
    gettimeofday(&start, NULL); 
    ios_base::sync_with_stdio(false); 

	fstream fin; 
    fin.open("/home/tayal/Downloads/IPSC_Project/mnist_train.csv", ios::in); 
    // fin.open("/home/tayal/Downloads/IPSC_Project/apparel-trainval.csv", ios::in); 
    

    vector<string> row; 
    string line, word; 
    vector<vector<double>> data;
  	int i=datasize;
    while (fin >> line && i--) { 
        stringstream s(line);
        vector<double> d; 
        while (getline(s, word, ',')) { 
            d.push_back(stod(word)); 
        } 
        data.push_back(d);
    }
    fin.close();

    vector<vector<double>> x_train,x_test,y_train,y_test;
    train_test_split(data,x_train,x_test,y_train,y_test);

    vector<int> layers={784,100,10};
    NeuralNetwork nn(layers,0.1,1);
    // for(auto a:nn.layers)cout<<a;
    // cout<<nn.weights[0][0][1];

    int train_datasize=floor(0.8*datasize);

    // cout<<"Training";
    int k;
    for(int i=0;i<epochs;i++)
    {
    	for(int j=0;j<=train_datasize-batchsize;j+=batchsize)
    	{
    		vector<vector<double>> x,y;
    		for(k=j;k<j+batchsize;k++)
			{x.push_back(x_train[k]);
			y.push_back(y_train[k]);}
			nn.train(x,y);

    	}
    }
    // cout<<k;
    // cout<<"\n";
    vector<vector<double>> result,actual;
    vector<vector<vector<double>>> inputs,outputs;
    nn.feed_foreward(x_train,result,inputs,outputs);
    actual = parallel_transpose(y_train);

    // cout<<result.size()<<result[0].size();
    // cout<<actual.size()<<actual[0].size();
    // print("Printing first 10 results");
    // for(int j=0;j<10;j++)
    // {	
    // 	cout<<endl<<endl<<endl;
    // 	double sum=0;
    // 	for(int i=0;i<10;i++)
    // 	{sum+=result[i][j];
    // 	cout<<"\nactual:"<<actual[i][j];
    // 	cout<<"\nresult:"<<result[i][j];}
    // 	 // cout<<"\nsum:"<<sum<<endl;
    // }
    vector<int> act;
    vector<int> res;
    act = argument_max(actual);
    res = argument_max(result);
    double n=act.size();
    double count = 0;
    for(int i=0;i<n;i++)
    {
    	if(act[i]==res[i])count++;
    }
    cout<<"Datasize: "<<datasize;
    cout<<"\nBatchsize: "<<batchsize;
    cout<<"\nEpochs: "<<epochs;
    cout<<"\n\nAccuracy: "<<count/n;

   	// for(auto row:nn.weights[0])
   	// 	for(auto col:row)
   	// 		cout<<col<<" ";

   	gettimeofday(&end, NULL); 			// time calculation final
    double time_taken; 
    time_taken = (end.tv_sec - start.tv_sec) * 1e6; 
    time_taken = (time_taken + (end.tv_usec - start.tv_usec)) * 1e-6; 
    cout << "\nTime taken by program is : " << fixed 
         << time_taken << setprecision(6); 
    cout << " sec" << endl; 

    
    return 0;
}
    