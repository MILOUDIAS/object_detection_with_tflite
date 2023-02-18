/*
 * Code de detection d'object par tensorflow lite sur linux et raspberry pi
 * */
#include <cstdio>
#include <fstream>
#include <iostream>
#include <ctime>
#include <cmath>
#include <string>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/model.h"

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/ocl.hpp>

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

int numThreads = 4;
static int g_Width = 300;
static int g_Height = 300;
int frame_count = 0;
time_t start_time = time(NULL);
std::vector<std::string> Labels;
std::unique_ptr<tflite::Interpreter> interpreter;

// Number of frames to capture
int num_frames = 30;

// Start and end times
time_t start, end;

static bool getFileContent( std::string fileName ) {

	// Open the File
	std::ifstream in(fileName.c_str());
	// Check if object is valid
	if(!in.is_open()) return false;

	std::string str;
	// Read the next line from File untill it reaches the end.
	while (std::getline(in, str))
	{
		// Line contains string of length > 0 then save it in vector
		if(str.size()>0) Labels.push_back(str);
	}
	// Close The File
	in.close();
	return true;
}

void detect_from_video( cv::Mat &src ) {

    cv::Mat image;
    int cam_width =src.cols;
    int cam_height=src.rows;

    // copy image to input as input tensor
    cv::resize(src, image, cv::Size(g_Width,g_Height));
    memcpy(interpreter->typed_input_tensor<uchar>(0), image.data, image.total() * image.elemSize());
    // std::copy( image.data, image.total(), interpreter->typed_input_tensor<uchar>(0));
    interpreter->SetAllowFp16PrecisionForFp32(true);
    interpreter->SetNumThreads(4);      //quad core

//        cout << "tensors size: " << interpreter->tensors_size() << "\n";
//        cout << "nodes size: " << interpreter->nodes_size() << "\n";
//        cout << "inputs: " << interpreter->inputs().size() << "\n";
//        cout << "input(0) name: " << interpreter->GetInputName(0) << "\n";
//        cout << "outputs: " << interpreter->outputs().size() << "\n";

    interpreter->Invoke();      // run your model

    const float* detection_locations = interpreter->tensor(interpreter->outputs()[0])->data.f;
    const float* detection_classes=interpreter->tensor(interpreter->outputs()[1])->data.f;
    const float* detection_scores = interpreter->tensor(interpreter->outputs()[2])->data.f;
    const int    num_detections = *interpreter->tensor(interpreter->outputs()[3])->data.f;

    //there are ALWAYS 10 detections no matter how many objects are detectable
//        cout << "number of detections: " << num_detections << "\n";
    const float confidence_threshold = 0.5;
    for(int i = 0; i < num_detections; i++){
        if(detection_scores[i] > confidence_threshold){
            int  det_index = (int)detection_classes[i]+1;
            float y1=detection_locations[4*i  ]*cam_height;
            float x1=detection_locations[4*i+1]*cam_width;
            float y2=detection_locations[4*i+2]*cam_height;
            float x2=detection_locations[4*i+3]*cam_width;

            cv::Rect rec((int)x1, (int)y1, (int)(x2 - x1), (int)(y2 - y1));
            rectangle(src,rec, cv::Scalar(0, 255, 0), 1, 8, 0);
            cv::putText(src, cv::format("%s", Labels[det_index].c_str()), cv::Point(x1, y1-5) ,cv::FONT_HERSHEY_SIMPLEX,0.5, cv::Scalar(0, 255, 0), 1, 8, 0);
        }
    }
}
int main( int argc,char * argv[] ) {

    if( argc < 2 ) {

        std::cerr << "Usage: " << argv[0] << " path/to/model.tflite path/to/labelsmap.txt\n";

        return 1;

    }

  const char* tflite_file = argv[1];
  const char* label_file = argv[2];

    cv::Mat frame;
    // Load model
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile( tflite_file );
    TFLITE_MINIMAL_CHECK( model != nullptr );
    // Build the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*model.get(), resolver)( &interpreter, numThreads );

    interpreter->AllocateTensors();

	// Get the names
	bool result = getFileContent( label_file );
	if(!result)
	{
        std::cout << "loading labels failed";
        exit(-1);
	}

    cv::VideoCapture cap(0);                 // Logitech Webcam
    if (!cap.isOpened()) {
        std::cerr << "ERROR: Unable to open the camera \n";
        return 0;
    }

    std::cout << "Start grabbing, press ESC on Live window to terminate \n";

    while( true ){

        cap >> frame;
        if (frame.empty()) {
            std::cerr << "ERROR: Unable to grab from the camera \n";
            break;
        }
        // increment the frame count
        frame_count++;
        // calculate the elapsed time
        time_t elapsed_time = time(NULL) - start_time;
        // calculate the FPS
        double fps = frame_count / (double)elapsed_time;

        detect_from_video(frame);

        cv::namedWindow("Name", cv::WINDOW_FULLSCREEN);

        cv::putText(frame, cv::format("FPS %0.2f",fps),cv::Point(10,20),cv::FONT_HERSHEY_SIMPLEX,0.6, cv::Scalar(0, 0, 255));
        //show output
        cv::imshow("Object Detection avec Tensorflow lite", frame);

        char esc = cv::waitKey(5);
        if(esc == 27) break;
    }

    std::cout << "Closing the camera \n";
    cv::destroyAllWindows();
    std::cout << "Bye! \n";


  return 0;
}
