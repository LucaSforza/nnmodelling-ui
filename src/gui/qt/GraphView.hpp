#ifndef NN_GRAPH_VIEW_HPP
#define NN_GRAPH_VIEW_HPP

#include <QGraphicsView>

class GraphScene;

class GraphView final : public QGraphicsView {
    Q_OBJECT
public:
    explicit GraphView(GraphScene *scene, QWidget *parent = nullptr);
    void fitGraph();

protected:
    void wheelEvent(QWheelEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;
    void keyPressEvent(QKeyEvent *event) override;
    void keyReleaseEvent(QKeyEvent *event) override;

private:
    bool spaceDown_ = false;
    bool panning_ = false;
    QPoint lastPan_;
    qreal zoom_ = 1.0;
};

#endif
