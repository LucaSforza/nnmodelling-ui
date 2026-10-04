#ifndef NN_GUI_NETWORK_3D_VIEW_HPP
#define NN_GUI_NETWORK_3D_VIEW_HPP

#include <QWidget>
#include <QRect>

#include "visualization/visualization.h"

class QLabel;
class QTimer;
class QToolButton;

class Network3DView final : public QWidget {
    Q_OBJECT
public:
    explicit Network3DView(QWidget *parent = nullptr);
    ~Network3DView() override;

    bool rebuild(const NNProject *project);
    QString selectedPath() const;
    QRect sceneViewportRect() const;

public slots:
    void fitWholeGraph();
    void homeCamera();

signals:
    void sceneReady(bool ready);

protected:
    void paintEvent(QPaintEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;
    void keyPressEvent(QKeyEvent *event) override;
    void keyReleaseEvent(QKeyEvent *event) override;
    void focusOutEvent(QFocusEvent *event) override;
    void resizeEvent(QResizeEvent *event) override;

private:
    void updateFrame();
    void refreshCameraAspect();
    void pickAt(const QPointF &position);
    void updateSelection(size_t nodeIndex);
    void tickMovement();

    NN3DScene *scene_ = nullptr;
    NN3DCamera camera_{};
    NN3DFrame frame_{};
    QLabel *selection_ = nullptr;
    QLabel *error_ = nullptr;
    QTimer *movementTimer_ = nullptr;
    QToolButton *fitButton_ = nullptr;
    QToolButton *homeButton_ = nullptr;
    QPoint lastMouse_;
    QString selectedPath_;
    bool rightDragging_ = false;
    bool keys_[6] = {};
};

#endif
