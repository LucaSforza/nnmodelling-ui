#ifndef NN_EDGE_ITEM_HPP
#define NN_EDGE_ITEM_HPP

#include <QGraphicsPathItem>
#include <QString>

class PortItem;

class EdgeItem final : public QGraphicsPathItem {
public:
    EdgeItem(QString id, PortItem *source, PortItem *target);
    QString id() const { return id_; }
    void updatePath();
private:
    QString id_;
    PortItem *source_;
    PortItem *target_;
};

#endif
