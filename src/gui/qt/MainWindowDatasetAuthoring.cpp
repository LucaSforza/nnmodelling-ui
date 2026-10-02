#include "MainWindow.hpp"
#include "MainWindowUtils.hpp"
#include "application/application.h"
#include <QCheckBox>
#include <QDialog>
#include <QDialogButtonBox>
#include <QFormLayout>
#include <QHeaderView>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QSet>
#include <QStatusBar>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

using namespace MainWindowUtils;

void MainWindow::createDataset() {
    if (!nn_app_project(application_.get())) {
        statusBar()->showMessage(tr("Open a project before creating a dataset"), 5000); return;
    }
    QDialog dialog(this);
    dialog.setWindowTitle(tr("Create dataset — saves project"));
    auto *layout = new QVBoxLayout(&dialog);
    auto *form = new QFormLayout;
    auto *id = new QLineEdit(&dialog); id->setObjectName("datasetId");
    auto *version = new QLineEdit(QStringLiteral("1.0.0"), &dialog); version->setObjectName("datasetVersion");
    auto *name = new QLineEdit(&dialog); name->setObjectName("datasetName");
    auto *description = new QLineEdit(&dialog); description->setObjectName("datasetDescription");
    auto *select = new QCheckBox(tr("Select this dataset after creation"), &dialog);
    select->setObjectName("datasetSelect"); select->setChecked(true);
    form->addRow(tr("ID"), id); form->addRow(tr("Version"), version);
    form->addRow(tr("Name"), name); form->addRow(tr("Description"), description);
    form->addRow(QString(), select); layout->addLayout(form);
    auto makeSlots = [&dialog, layout](const QString &title, const QString &objectName) {
    layout->addWidget(new QLabel(title + tr(" — shape dimensions comma separated (B allowed); dtypes: float16, bfloat16, float32/64, int8/16/32/64, uint8, bool."), &dialog));
        auto *table = new QTableWidget(0, 3, &dialog); table->setObjectName(objectName);
        table->setHorizontalHeaderLabels({tr("Slot name"), tr("Dtype"), tr("Shape")});
        table->horizontalHeader()->setStretchLastSection(true); layout->addWidget(table);
        auto *actions = new QHBoxLayout; auto *add = new QPushButton(tr("Add row"), &dialog);
        auto *remove = new QPushButton(tr("Remove row"), &dialog); actions->addWidget(add); actions->addWidget(remove); actions->addStretch(); layout->addLayout(actions);
        QObject::connect(add, &QPushButton::clicked, table, [table] { addTableRow(table, {QString(), QStringLiteral("float32"), QStringLiteral("B")}); });
        QObject::connect(remove, &QPushButton::clicked, table, [table] { if (table->currentRow() >= 0) table->removeRow(table->currentRow()); });
        return table;
    };
    auto *inputs = makeSlots(tr("Input slots"), QStringLiteral("datasetInputs"));
    auto *targets = makeSlots(tr("Target slots"), QStringLiteral("datasetTargets"));
    auto *formError = new QLabel(&dialog);
    formError->setObjectName(QStringLiteral("datasetError"));
    formError->setWordWrap(true);
    formError->setStyleSheet(QStringLiteral("color: #a12b2b;"));
    layout->addWidget(formError);
    auto *buttons = new QDialogButtonBox(QDialogButtonBox::Save | QDialogButtonBox::Cancel, &dialog);
    buttons->button(QDialogButtonBox::Save)->setText(tr("Create and save project"));
    connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject); layout->addWidget(buttons);
    while (dialog.exec() == QDialog::Accepted) {
    formError->clear();
    auto buildSlots = [this](QTableWidget *table, QJsonObject *out) -> bool {
        static const QSet<QString> dtypes = {"float16", "bfloat16", "float32", "float64", "int8", "int16", "int32", "int64", "uint8", "bool"};
        for (int row = 0; row < table->rowCount(); ++row) {
            const QString slot = cellText(table, row, 0), dtype = cellText(table, row, 1);
            if (slot.isEmpty() || out->contains(slot) || !dtypes.contains(dtype)) return false;
            QJsonArray shape;
            const QStringList dims = cellText(table, row, 2).split(',', Qt::KeepEmptyParts);
            if (dims.isEmpty() || dims.size() > 64) return false;
            for (QString dim : dims) {
                dim = dim.trimmed();
                if (dim == QStringLiteral("B")) { shape.append(dim); continue; }
                bool ok = false; const qlonglong n = dim.toLongLong(&ok);
                if (!ok || n <= 0) return false;
                shape.append(QJsonValue(qint64(n)));
            }
            QJsonObject tensor; tensor.insert("dtype", dtype); tensor.insert("shape", shape); out->insert(slot, tensor);
        }
        return true;
    };
    QJsonObject inputSlots, targetSlots;
    if (!buildSlots(inputs, &inputSlots) || inputSlots.isEmpty() || !buildSlots(targets, &targetSlots)) {
        formError->setText(tr("Input slots are required; use unique names, supported dtypes and positive dimensions (or B).")); continue;
    }
    QJsonObject batch; batch.insert("inputs", inputSlots); batch.insert("targets", targetSlots);
    QJsonObject definition; definition.insert("name", name->text().trimmed());
    definition.insert("description", description->text().trimmed()); definition.insert("batch", batch);
    const QByteArray idBytes = id->text().trimmed().toUtf8(), versionBytes = version->text().trimmed().toUtf8();
    const QByteArray json = QJsonDocument(definition).toJson(QJsonDocument::Compact);
    char error[ErrorCapacity] = {};
    if (!nn_app_create_dataset(application_.get(), idBytes.constData(), versionBytes.constData(),
                              json.constData(), select->isChecked(), error, sizeof(error))) {
        formError->setText(QString::fromUtf8(error)); continue;
    }
    refreshAll();
    statusBar()->showMessage(tr("Dataset created; current project saved"), 6000);
    break;
    }
}
