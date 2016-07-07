#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>

//toString template
template <typename T>
std::string ToString(T val)
{
    std::stringstream stream;
    stream << val;
    return stream.str();
}

/*
* Convert training/testing set to be used by OpenCV Machine Learning algorithms.
* TrainData is a matrix of size (#samples x max(#cols,#rows) per samples), in 32FC1.
*/
void ConvertToMl(const std::vector< cv::Mat > & train_samples, cv::Mat& train_data)
{
    //--Convert data
    const int rows = (int)train_samples.size();
    const int cols = std::max<int>(train_samples[0].cols, train_samples[0].rows);
    cv::Mat tmp(1, cols, CV_32FC1);
    train_data = cv::Mat(rows, cols, CV_32FC1);

    for (const cv::Mat& mat : train_samples) {

        size_t index = &mat - &train_samples[0];

        CV_Assert(mat.cols == 1 || mat.rows == 1);
        if (mat.cols == 1)
        {
            cv::transpose(mat, tmp);
            tmp.copyTo(train_data.row(index));
        }
        else if (mat.rows == 1)
        {
            mat.copyTo(train_data.row(index));
        }
    }
}

void ConvertToHogVevtor(int picture_index, cv::String file_path, std::vector< cv::Mat >& gradient_list, cv::Size size) {

    cv::Mat original_train_mat = cv::imread(file_path);
    cv::Mat taget_image2, mat_for_vector;

    cv::resize(original_train_mat, taget_image2, size, 0, 0, cv::INTER_CUBIC);
    cv::cvtColor(taget_image2, mat_for_vector, cv::COLOR_BGR2GRAY);
    cv::threshold(mat_for_vector, mat_for_vector, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    cv::HOGDescriptor hog;
    hog.winSize = size;
    std::vector<cv::Point> location;
    std::vector< float > descriptors;

    hog.compute(mat_for_vector, descriptors, cv::Size(8, 8), cv::Size(0, 0), location);

    gradient_list.push_back(cv::Mat(descriptors).clone());

}

int main()
{
    int picture_counts = 3;
    int picture_kinds = 4;
    cv::Size picture_size(8 * 4, 8 * 4);

    cv::String filename_KNearest = "KNearestDigit.xml";
    cv::String filename_SVM = "SVMDigit.xml";
    cv::String picture_path = "train/";
    cv::String picture_file = "";

    cv::Mat_<int> train_labels(1, picture_counts * picture_kinds);
    std::vector< cv::Mat > gradient_lst;
    std::vector< cv::Mat > gradient_test_lst;

    //KNearest
    cv::Ptr<cv::ml::KNearest>  knn(cv::ml::KNearest::create());

    //SVM
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
    svm->setType(cv::ml::SVM::C_SVC);
    svm->setKernel(cv::ml::SVM::LINEAR);
    svm->setGamma(3);
    svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 1e-6));

    cv::Mat train_data;
    int index = 0;

    std::cout << "KNearest SVM train.." << std::endl;
    for (int picture_kind = 1; picture_kind <= picture_kinds; picture_kind++) {
        for (int picture_number = 0; picture_number < picture_counts; picture_number++) {
            picture_file = picture_path + "img_" + ToString(picture_kind) + "_" + ToString(picture_number) + ".bmp";
            std::cout << "Picture:" << picture_file << std::endl;

            train_labels(0, index) = picture_kind;
            ConvertToHogVevtor(index++, picture_file, gradient_lst, picture_size);
        }
    }

    ConvertToMl(gradient_lst, train_data);

    // Train the KNearest
    knn->train(train_data, cv::ml::ROW_SAMPLE, train_labels);
    // Train the SVM
    svm->train(train_data, cv::ml::ROW_SAMPLE, train_labels);

    knn->save(filename_KNearest);
    svm->save(filename_SVM);

    // ============ TEST ============
    cv::Mat test_data;

    picture_file = picture_path + "img_2_0.bmp";
    ConvertToHogVevtor(index++, picture_file, gradient_test_lst, picture_size);
    ConvertToMl(gradient_test_lst, test_data);

    //KNearest
    int K = 1;
    cv::Mat response, dist;
    knn->findNearest(test_data, K, cv::noArray(), response, dist);
    std::cout << "KNearest ANS:" << response << std::endl;

    //SVM
    int response_svm = static_cast<int>(svm->predict(test_data));
    std::cout << "SVM ANS:" << response_svm << std::endl;

    system("pause");

    return 0;

}

