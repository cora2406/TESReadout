#include <iostream>
#include <Dense>
#include <chrono>
#include <unistd.h>

using namespace Eigen;
using namespace std;

#define FILTERS 4
#define VERBOSE 2
#define REPEAT 1
#define SAMPLESPEREVENT 100
#define MAXWINDOW 6
#define MAXSAMPLESIZE MAXWINDOW*SAMPLESPEREVENT
#define MAXFILTERS 5
#define NUMEVENTS 1000

float evaluate_energy(const ArrayXi int_samples, const ArrayXXf filters, int windowsize){

    int size = SAMPLESPEREVENT*windowsize;
    ArrayXf temp(size);
    ArrayXf samples = (int_samples.cast<float>() / 1000).head(size);
    float energy = (samples * filters.topLeftCorner(size,1)).sum();

    #if FILTERS > 1
    temp = 1/((samples + 20 * ArrayXf::Constant(size,52))); // linear transformation
    energy = energy + (temp * filters.block(0,1,size,1)).sum();
    #endif

    #if FILTERS > 2
    temp = 1/((samples + 40 * ArrayXf::Constant(size,25))); // linear transformation
    energy = energy + (temp * filters.block(0,2,size,1)).sum();
    #endif

    #if FILTERS > 3
    temp = 1/((samples - 32 * ArrayXf::Constant(size,0.45))); // linear transformation
    energy = energy + (temp * filters.block(0,3,size,1)).sum();
    #endif

    return energy;

}

int main(int argc, char** argv)
{

    ArrayXi samples = ArrayXi::Random(MAXSAMPLESIZE);
    ArrayXXf filters = ArrayXXf::Random(MAXSAMPLESIZE, FILTERS);
    ArrayXXf tempEvents = ArrayXXf::Random(1, NUMEVENTS);
    Array<bool, 1, NUMEVENTS> isEvent;

    isEvent = tempEvents > 0.5;

    int sum;

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int repeat = 0; repeat<REPEAT; repeat++){
        for (int i = 4; i<NUMEVENTS; i++){
            if (isEvent(i) == true){
                if (isEvent(i-1) == true){
                    //use filter of length 2
                    sum = evaluate_energy(samples, filters, 2);
                    #if VERBOSE > 1
                    cout << isEvent.segment<6>(i-5) << " -- " << "2 period filter used. The sum is " << sum << endl;
                    #endif
                }
                else{
                    if (isEvent(i-2) == true){
                    //use filter of length 3
                    sum = evaluate_energy(samples, filters, 3);
                    #if VERBOSE > 1
                    cout << isEvent.segment<6>(i-5) << " -- " << "3 period filter used. The sum is " << sum << endl;
                    #endif

                    }
                    else{
                        if (isEvent(i-3) == true){
                        //use filter of length 4
                        sum = evaluate_energy(samples, filters, 4);
                        #if VERBOSE > 1
                        cout << isEvent.segment<6>(i-5) << " -- " << "4 period filter used.  The sum is " << sum  << endl;
                        #endif
                        }
                        else{
                            if (isEvent(i-4) == true){
                                //use filter of length 5
                                sum = evaluate_energy(samples, filters, 5);
                                #if VERBOSE > 1
                                cout << isEvent.segment<6>(i-5) << " -- " << "5 period filter used. The sum is " << sum  << endl;
                                #endif
                            }
                            else{
                                //use single event filter
                                sum = evaluate_energy(samples, filters, 1);
                                #if VERBOSE > 1
                                cout << isEvent.segment<6>(i-5) << " -- " << "Single period filter used. The sum is " << sum  << endl;
                                #endif
                            }
                        }
                    }
                }
            }

        }
    }


    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> fp_ms = t2 - t1;

    #if VERBOSE > 0
    cout << "The sum is : " << sum << endl;
    cout << "The process took " << fp_ms.count() << "ms" << endl;
    #endif

  //Multiply vectors then sum the coefficients for inner product.

}
