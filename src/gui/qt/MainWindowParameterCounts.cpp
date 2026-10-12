#include "MainWindow.hpp"
#include "MainWindowUtils.hpp"

#include "application/application.h"

#include <QDialog>
#include <QDialogButtonBox>
#include <QHBoxLayout>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QMessageBox>
#include <QPushButton>
#include <QTableWidget>
#include <QHeaderView>
#include <QVBoxLayout>

using namespace MainWindowUtils;

namespace {
QString countText(const QJsonValue &value)
{
    return value.isDouble() ? QString::number(static_cast<qlonglong>(value.toInteger()))
                            : QStringLiteral("—");
}
}

void MainWindow::showParameterCounts()
{
    if (!nn_app_project(application_.get())) return;
    auto *dialog = new QDialog(this);
    dialog->setObjectName(QStringLiteral("parameterCountsDialog"));
    dialog->setWindowTitle(tr("Parameter counts"));
    dialog->resize(760, 480);
    auto *layout = new QVBoxLayout(dialog);
    auto *summary = new QLabel(dialog);
    summary->setObjectName(QStringLiteral("parameterCountSummary"));
    layout->addWidget(summary);
    auto *table = new QTableWidget(dialog);
    table->setObjectName(QStringLiteral("parameterCountRows"));
    table->setColumnCount(5);
    table->setHorizontalHeaderLabels({tr("Node"), tr("Status"), tr("Total"),
                                      tr("Trainable"), tr("Message")});
    table->setEditTriggers(QAbstractItemView::NoEditTriggers);
    table->setSelectionBehavior(QAbstractItemView::SelectRows);
    table->setSelectionMode(QAbstractItemView::SingleSelection);
    table->horizontalHeader()->setStretchLastSection(true);
    table->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    table->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    layout->addWidget(table, 1);
    auto *buttons = new QDialogButtonBox(QDialogButtonBox::Close, dialog);
    auto *refresh = buttons->addButton(tr("Refresh"), QDialogButtonBox::ActionRole);
    refresh->setObjectName(QStringLiteral("refreshParameterCounts"));
    layout->addWidget(buttons);
    QObject::connect(buttons, &QDialogButtonBox::rejected, dialog, &QDialog::reject);
    QObject::connect(buttons, &QDialogButtonBox::accepted, dialog, &QDialog::accept);
    const auto load = [this, dialog, summary, table] {
        char error[ErrorCapacity] = {};
        char *text = nn_app_parameter_counts_json(application_.get(), error, sizeof(error));
        if (!text) {
            QMessageBox::warning(dialog, tr("Parameter counts unavailable"), QString::fromUtf8(error));
            return;
        }
        const QJsonDocument document = QJsonDocument::fromJson(QByteArray(text));
        nn_app_free_text(text);
        if (!document.isObject()) {
            QMessageBox::warning(dialog, tr("Parameter counts unavailable"),
                                 tr("The parameter count report could not be read."));
            return;
        }
        const QJsonObject root = document.object();
        const bool complete = root.value(QStringLiteral("complete")).toBool();
        summary->setText(tr("Total parameters: %1    Trainable: %2    %3")
            .arg(countText(root.value(QStringLiteral("total"))),
                 countText(root.value(QStringLiteral("trainable"))),
                 complete ? tr("Complete") : tr("Incomplete; unknown contributions are shown below")));
        const QJsonArray rows = root.value(QStringLiteral("rows")).toArray();
        table->setRowCount(rows.size());
        for (qsizetype i = 0; i < rows.size(); ++i) {
            const QJsonObject row = rows.at(i).toObject();
            const QStringList cells = {
                row.value(QStringLiteral("node_id")).toString(),
                row.value(QStringLiteral("status")).toString(),
                countText(row.value(QStringLiteral("total"))),
                countText(row.value(QStringLiteral("trainable"))),
                row.value(QStringLiteral("message")).toString()
            };
            for (int column = 0; column < cells.size(); ++column)
                table->setItem(static_cast<int>(i), column, new QTableWidgetItem(cells[column]));
        }
        table->resizeRowsToContents();
    };
    QObject::connect(refresh, &QPushButton::clicked, dialog, load);
    load();
    dialog->exec();
    // Keep the modal and its children alive through Qt's queued redraw after close.
    dialog->deleteLater();
}
