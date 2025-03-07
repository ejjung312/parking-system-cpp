#include <Eigen/Dense>

class KalmanFilter {
private:
    Eigen::VectorXd x_;
    Eigen::MatrixXd F_, H_, P_, Q_, R_;

public:
    KalmanFilter() {
        int stateSize = 4; // x, y, vx, vy
        int measSize = 2; // x, y
        int controlSize = 0;

        x_ = Eigen::VectorXd(stateSize);
        F_ = Eigen::MatrixXd::Identity(stateSize, stateSize);
        H_ = Eigen::MatrixXd::Zero(measSize, stateSize);
        P_ = Eigen::MatrixXd::Identity(stateSize, stateSize) * 1000;
        Q_ = Eigen::MatrixXd::Identity(stateSize, stateSize) * 1e-2;
        R_ = Eigen::MatrixXd::Identity(measSize, measSize) * 1e-1;

        H_.block(0, 0, measSize, measSize) = Eigen::MatrixXd::Identity(measSize, measSize);
    }

    void predict() {
        x_ = F_ * x_;
        P_ = F_ * P_ * F_.transpose() + Q_;
    }

    void update(const Eigen::VectorXd& z) {
        Eigen::VectorXd y = z - H_ * x_;
        Eigen::MatrixXd S = H_ * P_ * H_.transpose() + R_;
        Eigen::MatrixXd K = P_ * H_.transpose() * S.inverse();

        x_ = x_ + K * y;
        P_ = (Eigen::MatrixXd::Identity(x_.size(), x_.size()) - K * H_) * P_;
    }

    Eigen::VectorXd getState() { return x_; }

    void setState(const Eigen::VectorXd& state) { x_ = state; }
};