#ifndef NN_PORT_ITEM_HPP
#define NN_PORT_ITEM_HPP

#include <QGraphicsObject>
#include <QPainterPath>
#include <QString>

class GraphScene;

class PortItem final : public QGraphicsObject {
    Q_OBJECT
public:
    PortItem(GraphScene *owner, QString nodeId, QString handleId, bool output,
             QString label, QString outputType = {}, QGraphicsItem *parent = nullptr);
    QRectF boundingRect() const override;
    QPainterPath shape() const override;
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
               QWidget *widget = nullptr) override;
    QString nodeId() const { return nodeId_; }
    QString handleId() const { return handleId_; }
    QString label() const { return label_; }
    bool isOutput() const { return output_; }
    QString outputType() const { return outputType_; }
    void setBoundaryHandlePresentation(bool boundary) {
        boundaryHandlePresentation_ = boundary;
        update();
    }

protected:
    void mousePressEvent(QGraphicsSceneMouseEvent *event) override;
    void mouseMoveEvent(QGraphicsSceneMouseEvent *event) override;
    void mouseReleaseEvent(QGraphicsSceneMouseEvent *event) override;
    void hoverEnterEvent(QGraphicsSceneHoverEvent *event) override;
    void hoverLeaveEvent(QGraphicsSceneHoverEvent *event) override;

private:
    GraphScene *owner_;
    QString nodeId_;
    QString handleId_;
    QString label_;
    QString outputType_;
    bool output_;
    bool hovered_ = false;
    bool boundaryHandlePresentation_ = false;
};

#endif
