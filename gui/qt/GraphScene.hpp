#ifndef NN_GRAPH_SCENE_HPP
#define NN_GRAPH_SCENE_HPP

#include <QGraphicsScene>
#include <QHash>
#include <QColor>
#include <QPointF>
#include <QString>
#include <QStringList>

#include "application.h"

class EdgeItem;
class NodeItem;
class PortItem;
class QGraphicsPathItem;

class GraphScene final : public QGraphicsScene {
    Q_OBJECT
public:
    explicit GraphScene(NNApplication *application, QObject *parent = nullptr);

    void refresh();
    void setScope(const QString &scopeId);
    QString scope() const { return scopeId_; }
    void deleteSelection();
    void goToParentScope();
    void cancelInteraction();
    QString selectedNodeId() const;
    NodeItem *nodeItem(const QString &id) const;
    EdgeItem *edgeItem(const QString &id) const;
    void setProblemMarkers(const QHash<QString, QString> &categories);
    void revealNode(const QString &id);
    QColor connectionDraftColor() const;

signals:
    void modelChanged();
    void scopeChanged(const QString &scopeId);
    void errorOccurred(const QString &message);

protected:
    void keyPressEvent(QKeyEvent *event) override;

private:
    friend class NodeItem;
    friend class PortItem;
    void updateEdgesForNode(const QString &nodeId);
    void beginNodeDrag(NodeItem *item);
    void commitNodeMoves();
    void cancelNodeDrag();
    void scheduleRefresh();
    void beginConnection(PortItem *port);
    void updateConnection(const QPointF &position);
    void finishConnection(PortItem *port);
    void cancelConnection();
    void reportError(const char *message);
    QString parentScope() const;

    NNApplication *application_;
    QString scopeId_;
    QHash<QString, NodeItem *> nodes_;
    QHash<QString, EdgeItem *> edges_;
    QHash<QString, QString> problemCategories_;
    PortItem *draftSource_ = nullptr;
    QGraphicsPathItem *draftPath_ = nullptr;
    bool refreshing_ = false;
    bool refreshPending_ = false;
    QHash<QString, QPointF> dragPositions_;
    QStringList lockedMovableNodes_;
};

#endif
