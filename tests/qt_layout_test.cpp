#include "DataflowLayout.hpp"
#include "model/model.h"

#include <QTest>

class DataflowLayoutTest final : public QObject {
    Q_OBJECT
private slots:
    void alignsChainsAndSeparatesBranchesAndJoins();
};

void DataflowLayoutTest::alignsChainsAndSeparatesBranchesAndJoins() {
    const QVector<DataflowLayout::Node> nodes = {
        {"source", QRectF(0, 0, 190, 190), 47, false},
        {"branch-a", QRectF(0, 0, 190, 190), 95, false},
        {"branch-b", QRectF(0, 0, 270, 270), 135, false},
        {"join", QRectF(0, 0, 300, 300), 150, false},
        {"expanded-owner", QRectF(0, 0, 560, 560), 280, false},
        {"output", QRectF(0, 0, 190, 190), 47, true},
    };
    const QVector<DataflowLayout::Edge> edges = {
        {"source", "branch-a"}, {"source", "branch-b"},
        {"branch-a", "join"}, {"branch-b", "join"},
        {"join", "expanded-owner"}, {"expanded-owner", "output"},
    };

    for (FlowDirection direction : {FlowDirection::Vertical, FlowDirection::Horizontal}) {
        const auto positions = DataflowLayout::arrange(nodes, edges, direction);
        QCOMPARE(positions.size(), nodes.size());
        auto crossAnchor = [&](const char *id, qreal local) {
            const QPointF origin = positions.value(QString::fromLatin1(id));
            return (direction == FlowDirection::Vertical ? origin.x() : origin.y()) + local;
        };
        const qreal source = crossAnchor("source", 47);
        const qreal branchMean =
            (crossAnchor("branch-a", 95) + crossAnchor("branch-b", 135)) / 2.0;
        QVERIFY(qAbs(source - branchMean) <= NN_MODEL_GRID_SPACING);
        const qreal joinExpected =
            (crossAnchor("branch-a", 95) + crossAnchor("branch-b", 135)) / 2.0;
        QVERIFY(qAbs(crossAnchor("join", 150) - joinExpected) <= NN_MODEL_GRID_SPACING);
        QVERIFY(qAbs(crossAnchor("join", 150) -
                     crossAnchor("expanded-owner", 280)) <= NN_MODEL_GRID_SPACING);

        QHash<QString, QRectF> boxes;
        for (int i = 0; i < nodes.size(); ++i) {
            const QPointF origin = positions.value(nodes[i].id);
            QCOMPARE(qint32(origin.x()) % NN_MODEL_GRID_SPACING, 0);
            QCOMPARE(qint32(origin.y()) % NN_MODEL_GRID_SPACING, 0);
            const QRectF rect = nodes[i].bounds.translated(origin);
            boxes.insert(nodes[i].id, rect);
            for (int j = i + 1; j < nodes.size(); ++j) {
                const QRectF other = nodes[j].bounds.translated(positions.value(nodes[j].id));
                QVERIFY(!rect.intersects(other));
            }
        }
        for (const DataflowLayout::Edge &edge : edges) {
            const QRectF sourceRect = boxes.value(edge.source);
            const QRectF targetRect = boxes.value(edge.target);
            const qreal gap = direction == FlowDirection::Vertical
                ? targetRect.top() - sourceRect.bottom()
                : sourceRect.left() - targetRect.right();
            QVERIFY(gap >= 60.0);
        }
    }
}

QTEST_MAIN(DataflowLayoutTest)
#include "qt_layout_test.moc"
