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
    int first_index=0;
    int last_index=100;
    ArrayXXf tempEvents = ArrayXXf::Random(1, NUMEVENTS);
    Array<bool, 1, NUMEVENTS> isEvent;

    isEvent = tempEvents > 0.5;

    int sum=0;

    auto t1 = std::chrono::high_resolution_clock::now();
//    for (int repeat = 0; repeat<REPEAT; repeat++){
        read_index = 3*SAMPLESPERWINDOW;
        for (int i = 3; i<NUMEVENTS-1; i++){
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
                if (isEvent(i+1) == true){
                    //use filter of group A - ignore following window
                    selectedFilters = filterGroupA;
                    last_index = read_index+SAMPLESPERWINDOW;
                }
                else{
                    //use filter of group B, include following window
                    selectedFilters = filterGroupB;
                    last_index = read_index+2*SAMPLESPERWINDOW;
                }

                if (isEvent(i-1) == true){
                    //Signal in previous window, include this window
                    first_index = read_index-SAMPLESPERWINDOW;
                }
                else{
                    //No signal in previous window, use window before that, different filter if there is a signal or not
                    first_index = read_index-2*SAMPLESPERWINDOW;
                    if (isEvent(i-2) == true){
                    }
                    else{
                    }
                }
            cout << isEvent.segment<5>(i-3) << " -- " << first_index << "-" << last_index << endl;
            }
            read_index += 100;
            if (read_index>samples.size()){
                read_index=0;
                wraparound=false;
            }

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
