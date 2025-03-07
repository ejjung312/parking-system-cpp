#include <opencv2/opencv.hpp>
#include <filesystem>

#include "yolo11.hpp"

// struct Detection {
//     float x1, y1, x2, y2, score;
// };

std::set<int> vehicles = {2,3,5,7}; // 차량 클래스 ID 목록

int main() {
    //std::string resource_dir = "./resources/";

    std::string resource_dir = "C:\\cpplus\\parking-system\\resources\\";

    std::string model_path = resource_dir + "yolo11n.onnx";
    std::string license_model_path = resource_dir + "license_plate_best.onnx";
    std::string input_path = resource_dir + "car2.mp4";

    std::string output_path = input_path.substr(0, input_path.find_last_of(".")) + "_out.mp4";

    cv::VideoCapture cap(input_path);
    assert(cap.isOpened() && "Error: Cannot open video file");

    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);

    cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('m','p','4','v'), fps, cv::Size(width, height));
    assert(writer.isOpened() && "Error: Cannot open output video file");

    Yolo11 model(model_path, 0.45f, 0.45f,
        [](int lbl_id, const std::string lbl)
        { return lbl_id >= 0 && lbl_id <= 8;} );

    cv::Mat frame;
    while (cap.read(frame)) {
        std::vector<ObjectBBox> bbox_l = model.detect(frame);

        std::vector<ObjectBBox> detections_;
        for (auto& bbox: bbox_l) {
            // bbox.label, bbox.conf, bbox.x1, bbox.y1, bbox.x2, bbox.y2
            if (vehicles.find(bbox.class_id) != vehicles.end()) {
                detections_.push_back(bbox);
            }
            // std::cout << "Label:" << bbox.label << " Conf: " << bbox.conf;
            // std::cout << "(" << bbox.x1 << ", " << bbox.y1 << ") ";
            // std::cout << "(" << bbox.x2 << ", " << bbox.y2 << ")" << std::endl;
            //
            // bbox.draw(frame);
        }

        // track vehicles

        // detect license plates

        writer.write(frame);

        char key = cv::waitKey(1);
        if (key == 27 || key == 'q') break;
    }

    cap.release();
    writer.release();
    cv::destroyAllWindows();

    std::cout << "Video saved as: " << output_path << std::endl;

    return 0;
}
