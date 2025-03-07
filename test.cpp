#include <opencv2/opencv.hpp>
#include <filesystem>

int main2() {
    std::filesystem::path currentPath = std::filesystem::current_path();
    std::cout << "currnet path: " << currentPath << std::endl;

    std::string imagePath = "./resources/input.jpg";
    cv::Mat image = cv::imread(imagePath);

    if (image.empty()) {
        std::cerr << "Failed to load image" << std::endl;
        return -1;
    }

    cv::rectangle(image, cv::Point(50, 50), cv::Point(200,200), cv::Scalar(255,0,0), 2);

    std::string outputPath = "./resources/output.jpg";
    cv::imwrite(outputPath, image);

    // std::string image = cv::imread();

    // std::string file_name = "/app/resources/input.jpg";
    // std::string file_name = "input.jpg";
    // cv::Mat image;
    // image = cv::imread("input.jpg");
    // cv::imread("./resources/input.jpg", cv::IMREAD_GRAYSCALE);

    return 0;
    // cv::imwrite("./resources/input.jpg", );
}