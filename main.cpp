#include <iostream>
#include <Dense>
#include <deque>
#include <chrono>
#include <unistd.h>

using namespace Eigen;
using namespace std;

#define FILTERS 4
#define VERBOSE 2
#define REPEAT 3
#define SAMPLESPERWINDOW 100
#define MAXWINDOW 100
#define MAXSAMPLESIZE MAXWINDOW*SAMPLESPERWINDOW
#define MAXFILTERS 3
#define NUMEVENTS 1000

float evaluate_energy(const ArrayXi int_samples, const ArrayXXf filters, int windowsize){

    int size = SAMPLESPERWINDOW*windowsize;
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

int make_data(ArrayXi *samples, int new_samples = -1, int start_index=0){
    //fills the array with random data. can specify width
    // returns new write index
        if (new_samples == -1){
        new_samples = (*samples).size();
        }

        if (start_index+new_samples >= (*samples).size()){
            int partial_samples = (*samples).size()-start_index;
            (*samples).block(start_index,0,partial_samples,1) = ArrayXi::Random(partial_samples, 1);
            partial_samples = new_samples-partial_samples;
            (*samples).block(start_index,0,partial_samples,1) = ArrayXi::Random(partial_samples, 1);
        }
        else {
            (*samples).block(start_index,0,new_samples,1) = ArrayXi::Random(new_samples, 1);
        }

        return (start_index+new_samples)%((*samples).size());
}

int main(int argc, char** argv)
{
    ArrayXi samples(MAXSAMPLESIZE);
    int write_index = make_data(&samples);
    int read_index=0;
    bool wraparound=true;
    ArrayXXf filterGroupA = ArrayXXf::Random(MAXSAMPLESIZE, FILTERS);
    ArrayXXf filterGroupB = ArrayXXf::Random(MAXSAMPLESIZE, FILTERS);
    ArrayXXf selectedFilters(MAXSAMPLESIZE, FILTERS);
    ArrayXXf tempEvents = ArrayXXf::Random(1, NUMEVENTS);
    Array<bool, 1, NUMEVENTS> isEvent;

    isEvent = tempEvents > 0.5;

    int sum=0;

    auto t1 = std::chrono::high_resolution_clock::now();
//    for (int repeat = 0; repeat<REPEAT; repeat++){
        read_index = 3*SAMPLESPERWINDOW;
        for (int i = 3; i<NUMEVENTS; i++){
            if (wraparound == false && (write_index - read_index <= 2*SAMPLESPERWINDOW)){
                write_index = make_data(&samples, (MAXWINDOW/2) * SAMPLESPERWINDOW, write_index);
                cout << "Index updated to : " << write_index << endl;
                if (read_index > write_index) { wraparound = true; }
            }
            if (wraparound == true && (samples.size()-(read_index - write_index) <= 2*SAMPLESPERWINDOW)){
                write_index = make_data(&samples, (MAXWINDOW/2) * SAMPLESPERWINDOW, write_index);
                cout << "Index updated to : " << write_index << endl;
            }
            if (isEvent(i) == true){
//                if (isEvent(i-1) == true){
//                    //use filter of length 2
//                    sum = evaluate_energy(samples, filters, 2);
//                    #if VERBOSE > 1
//                    cout << isEvent.segment<6>(i-5) << " -- " << "2 period filter used. The sum is " << sum << endl;
//                    #endif
//                }
//                else{
//                    if (isEvent(i-2) == true){
//                    //use filter of length 3
//                    sum = evaluate_energy(samples, filters, 3);
//                    #if VERBOSE > 1
//                    cout << isEvent.segment<6>(i-5) << " -- " << "3 period filter used. The sum is " << sum << endl;
//                    #endif
//
//                    }
//                    else{
//                        if (isEvent(i-3) == true){
//                        //use filter of length 4
//                        sum = evaluate_energy(samples, filters, 4);
//                        #if VERBOSE > 1
//                        cout << isEvent.segment<6>(i-5) << " -- " << "4 period filter used.  The sum is " << sum  << endl;
//                        #endif
//                        }
//                        else{
//                            if (isEvent(i-4) == true){
//                                //use filter of length 5
//                                sum = evaluate_energy(samples, filters, 5);
//                                #if VERBOSE > 1
//                                cout << isEvent.segment<6>(i-5) << " -- " << "5 period filter used. The sum is " << sum  << endl;
//                                #endif
//                            }
//                            else{
//                                //use single event filter
//                                sum = evaluate_energy(samples, filters, 1);
//                                #if VERBOSE > 1
//                                cout << isEvent.segment<6>(i-5) << " -- " << "Single period filter used. The sum is " << sum  << endl;
//                                #endif
//                            }
//                        }
//                    }
//                }
            }
            read_index += 100;
            if (read_index>samples.size()){
                read_index=0;
                wraparound=false;
            }
            cout << read_index << endl;
        }
 //   }


    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> fp_ms = t2 - t1;

    #if VERBOSE > 0
    cout << "The sum is : " << sum << endl;
    cout << "The process took " << fp_ms.count() << "ms" << endl;
    #endif

  //Multiply vectors then sum the coefficients for inner product.

}
