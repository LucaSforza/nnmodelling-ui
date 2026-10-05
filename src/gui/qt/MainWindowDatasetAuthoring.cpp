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
#include <QListWidget>
#include <QPushButton>
#include <QScrollArea>
#include <QSet>
#include <QStatusBar>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QMessageBox>

using namespace MainWindowUtils;

void MainWindow::manageDatasets() {
    if (!nn_app_project(application_.get())) {
        statusBar()->showMessage(tr("Open a project before managing datasets"), 5000);
        return;
    }

    QDialog dialog(this);
    dialog.setObjectName(QStringLiteral("datasetManager"));
    dialog.setWindowTitle(tr("Project datasets"));
    dialog.resize(620, 430);
    auto *layout = new QVBoxLayout(&dialog);
    auto *list = new QListWidget(&dialog);
    list->setObjectName(QStringLiteral("datasetManagerList"));
    list->setSelectionMode(QAbstractItemView::SingleSelection);
    layout->addWidget(list, 1);

    auto *actions = new QHBoxLayout;
    auto *newButton = new QPushButton(tr("New dataset"), &dialog);
    newButton->setObjectName(QStringLiteral("datasetManagerNew"));
    auto *editButton = new QPushButton(tr("Edit"), &dialog);
    editButton->setObjectName(QStringLiteral("datasetManagerEdit"));
    auto *selectButton = new QPushButton(tr("Select"), &dialog);
    selectButton->setObjectName(QStringLiteral("datasetManagerSelect"));
    auto *closeButton = new QPushButton(tr("Close"), &dialog);
    actions->addWidget(newButton);
    actions->addWidget(editButton);
    actions->addWidget(selectButton);
    actions->addStretch();
    actions->addWidget(closeButton);
    layout->addLayout(actions);

    auto refreshList = [this, list, editButton, selectButton](const QString &preferredIdentity) {
        list->clear();
        const NNProject *project = nn_app_project(application_.get());
        const NNDataset *active = project ? nn_project_active_dataset(project) : nullptr;
        for (size_t i = 0; project && i < nn_project_dataset_count(project); ++i) {
            const NNDataset *dataset = nn_project_dataset_at(project, i);
            if (!dataset || !dataset->id || !dataset->version) continue;
            const QString id = QString::fromUtf8(dataset->id);
            const QString version = QString::fromUtf8(dataset->version);
            const QString identity = QStringLiteral("%1@%2").arg(id, version);
            QString label = identity;
            if (dataset->name && *dataset->name)
                label = QStringLiteral("%1  —  %2").arg(identity, QString::fromUtf8(dataset->name));
            if (dataset == active) label = QStringLiteral("✓  %1  (%2)").arg(label, tr("active"));
            auto *item = new QListWidgetItem(label, list);
            item->setData(Qt::UserRole, id);
            item->setData(Qt::UserRole + 1, version);
            item->setData(Qt::UserRole + 2, dataset == active);
            if (identity == preferredIdentity) list->setCurrentItem(item);
        }
        editButton->setEnabled(list->currentItem() != nullptr);
        selectButton->setEnabled(list->currentItem() != nullptr &&
            !list->currentItem()->data(Qt::UserRole + 2).toBool());
    };
    auto updateButtons = [list, editButton, selectButton] {
        const auto *item = list->currentItem();
        editButton->setEnabled(item != nullptr);
        selectButton->setEnabled(item && !item->data(Qt::UserRole + 2).toBool());
    };
    QObject::connect(list, &QListWidget::currentItemChanged, &dialog,
                     [updateButtons] { updateButtons(); });
    QObject::connect(newButton, &QPushButton::clicked, &dialog, [this, &dialog, &refreshList] {
        createDataset();
        refreshList(QString());
        dialog.raise();
        dialog.activateWindow();
    });
    QObject::connect(editButton, &QPushButton::clicked, &dialog,
                     [this, list, &dialog, &refreshList] {
        const auto *item = list->currentItem();
        if (!item) return;
        const QString identity = QStringLiteral("%1@%2").arg(
            item->data(Qt::UserRole).toString(), item->data(Qt::UserRole + 1).toString());
        createDataset(item->data(Qt::UserRole).toString(), item->data(Qt::UserRole + 1).toString());
        refreshList(identity);
        dialog.raise();
        dialog.activateWindow();
    });
    QObject::connect(selectButton, &QPushButton::clicked, &dialog,
                     [this, list, &refreshList] {
        const auto *item = list->currentItem();
        if (!item) return;
        const QString identity = QStringLiteral("%1@%2").arg(
            item->data(Qt::UserRole).toString(), item->data(Qt::UserRole + 1).toString());
        const QByteArray id = item->data(Qt::UserRole).toString().toUtf8();
        const QByteArray version = item->data(Qt::UserRole + 1).toString().toUtf8();
        char error[ErrorCapacity] = {};
        if (!nn_app_select_dataset(application_.get(), id.constData(), version.constData(),
                                   error, sizeof(error))) {
            QMessageBox::warning(this, tr("Dataset selection failed"), QString::fromUtf8(error));
            return;
        }
        refreshAll();
        refreshList(identity);
    });
    QObject::connect(closeButton, &QPushButton::clicked, &dialog, &QDialog::reject);
    QObject::connect(list, &QListWidget::itemActivated, &dialog,
                     [editButton] { editButton->click(); });
    refreshList(QString());
    dialog.exec();
}

void MainWindow::createDataset(const QString &editId, const QString &editVersion) {
    if (!nn_app_project(application_.get())) {
        statusBar()->showMessage(tr("Open a project before creating a dataset"), 5000);
        return;
    }
    const bool editing = !editId.isEmpty() && !editVersion.isEmpty();
    QJsonObject originalDefinition;
    if (editing) {
        const QByteArray id = editId.toUtf8();
        const QByteArray version = editVersion.toUtf8();
        char error[ErrorCapacity] = {};
        char *owned = nn_app_dataset_definition(application_.get(), id.constData(), version.constData(),
                                                error, sizeof(error));
        if (!owned) {
            QMessageBox::warning(this, tr("Dataset could not be opened"), QString::fromUtf8(error));
            return;
        }
        QJsonParseError parseError{};
        const QJsonDocument document = QJsonDocument::fromJson(QByteArray(owned), &parseError);
        nn_app_free_text(owned);
        if (parseError.error != QJsonParseError::NoError || !document.isObject()) {
            QMessageBox::warning(this, tr("Dataset could not be opened"),
                                 tr("The dataset definition is malformed."));
            return;
        }
        originalDefinition = document.object();
    }

    QDialog dialog(this);
    dialog.setObjectName(QStringLiteral("datasetEditorDialog"));
    dialog.setWindowTitle(editing ? tr("Edit dataset — saves project")
                                  : tr("Create dataset — saves project"));
    dialog.resize(760, 620);
    auto *layout = new QVBoxLayout(&dialog);
    auto *scroll = new QScrollArea(&dialog);
    scroll->setWidgetResizable(true);
    auto *content = new QWidget(scroll);
    content->setObjectName(QStringLiteral("datasetEditorContent"));
    scroll->setStyleSheet(QStringLiteral(
        "QScrollArea { background: #eceff3; }"
        "QWidget#datasetEditorContent { background: #eceff3; color: #263446; }"));
    auto *contentLayout = new QVBoxLayout(content);
    auto *form = new QFormLayout;
    auto *id = new QLineEdit(editing ? editId : QString(), content);
    id->setObjectName(QStringLiteral("datasetId"));
    auto *version = new QLineEdit(editing ? editVersion : QStringLiteral("1.0.0"), content);
    version->setObjectName(QStringLiteral("datasetVersion"));
    if (editing) {
        id->setReadOnly(true);
        version->setReadOnly(true);
    }
    auto *name = new QLineEdit(content);
    name->setObjectName(QStringLiteral("datasetName"));
    auto *description = new QLineEdit(content);
    description->setObjectName(QStringLiteral("datasetDescription"));
    const QJsonObject initial = originalDefinition;
    name->setText(initial.value(QStringLiteral("name")).toString());
    description->setText(initial.value(QStringLiteral("description")).toString());
    form->addRow(tr("ID"), id);
    form->addRow(tr("Version"), version);
    form->addRow(tr("Name"), name);
    form->addRow(tr("Description"), description);
    auto *select = new QCheckBox(tr("Select this dataset after creation"), content);
    select->setObjectName(QStringLiteral("datasetSelect"));
    select->setChecked(true);
    if (editing) select->hide();
    form->addRow(QString(), select);
    contentLayout->addLayout(form);

    auto makeSlots = [content, contentLayout](const QString &title, const QString &objectName) {
        auto *heading = new QLabel(title + QObject::tr(" — shape dimensions comma separated (B allowed); dtypes: float16, bfloat16, float32/64, int8/16/32/64, uint8, bool."), content);
        heading->setWordWrap(true);
        contentLayout->addWidget(heading);
        auto *table = new QTableWidget(0, 3, content);
        table->setObjectName(objectName);
        table->setHorizontalHeaderLabels({QObject::tr("Slot name"), QObject::tr("Dtype"), QObject::tr("Shape")});
        table->horizontalHeader()->setStretchLastSection(true);
        table->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
        contentLayout->addWidget(table);
        auto *rowActions = new QHBoxLayout;
        auto *add = new QPushButton(QObject::tr("Add row"), content);
        auto *remove = new QPushButton(QObject::tr("Remove row"), content);
        rowActions->addWidget(add);
        rowActions->addWidget(remove);
        rowActions->addStretch();
        contentLayout->addLayout(rowActions);
        QObject::connect(add, &QPushButton::clicked, table, [table] {
            addTableRow(table, {QString(), QStringLiteral("float32"), QStringLiteral("B")});
        });
        QObject::connect(remove, &QPushButton::clicked, table, [table] {
            if (table->currentRow() >= 0) table->removeRow(table->currentRow());
        });
        return table;
    };
    auto *inputs = makeSlots(tr("Input slots"), QStringLiteral("datasetInputs"));
    auto *targets = makeSlots(tr("Target slots"), QStringLiteral("datasetTargets"));
    auto populateSlots = [](QTableWidget *table, const QJsonObject &slotDefinitions) {
        for (auto it = slotDefinitions.begin(); it != slotDefinitions.end(); ++it) {
            const QJsonObject tensor = it.value().toObject();
            QStringList dimensions;
            for (const QJsonValue &dimension : tensor.value(QStringLiteral("shape")).toArray())
                dimensions.append(dimension.isString() ? dimension.toString()
                                                       : QString::number(dimension.toInteger()));
            addTableRow(table, {it.key(), tensor.value(QStringLiteral("dtype")).toString(),
                                dimensions.join(QStringLiteral(", "))});
        }
    };
    const QJsonObject batch = initial.value(QStringLiteral("batch")).toObject();
    populateSlots(inputs, batch.value(QStringLiteral("inputs")).toObject());
    populateSlots(targets, batch.value(QStringLiteral("targets")).toObject());
    auto *formError = new QLabel(&dialog);
    formError->setObjectName(QStringLiteral("datasetError"));
    formError->setWordWrap(true);
    formError->setStyleSheet(QStringLiteral("color: #a12b2b;"));
    auto *buttons = new QDialogButtonBox(QDialogButtonBox::Save | QDialogButtonBox::Cancel, &dialog);
    buttons->button(QDialogButtonBox::Save)->setText(editing ? tr("Save changes")
                                                            : tr("Create and save project"));
    QObject::connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    QObject::connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    scroll->setWidget(content);
    layout->addWidget(scroll, 1);
    layout->addWidget(formError);
    layout->addWidget(buttons);

    while (dialog.exec() == QDialog::Accepted) {
        formError->clear();
        auto buildSlots = [this](QTableWidget *table, QJsonObject *out) -> bool {
            static const QSet<QString> dtypes = {"float16", "bfloat16", "float32", "float64", "int8", "int16", "int32", "int64", "uint8", "bool"};
            for (int row = 0; row < table->rowCount(); ++row) {
                const QString slot = cellText(table, row, 0).trimmed();
                const QString dtype = cellText(table, row, 1).trimmed();
                if (slot.isEmpty() || out->contains(slot) || !dtypes.contains(dtype)) return false;
                QJsonArray shape;
                const QStringList dims = cellText(table, row, 2).split(',', Qt::KeepEmptyParts);
                if (dims.isEmpty() || dims.size() > 64) return false;
                for (QString dim : dims) {
                    dim = dim.trimmed();
                    if (dim == QStringLiteral("B")) { shape.append(dim); continue; }
                    bool ok = false;
                    const qlonglong n = dim.toLongLong(&ok);
                    if (!ok || n <= 0) return false;
                    shape.append(QJsonValue(qint64(n)));
                }
                QJsonObject tensor;
                tensor.insert(QStringLiteral("dtype"), dtype);
                tensor.insert(QStringLiteral("shape"), shape);
                out->insert(slot, tensor);
            }
            return true;
        };
        QJsonObject inputSlots, targetSlots;
        if (!buildSlots(inputs, &inputSlots) || inputSlots.isEmpty() || !buildSlots(targets, &targetSlots)) {
            formError->setText(tr("Input slots are required; use unique names, supported dtypes and positive dimensions (or B)."));
            continue;
        }
        QJsonObject batchDefinition;
        batchDefinition.insert(QStringLiteral("inputs"), inputSlots);
        batchDefinition.insert(QStringLiteral("targets"), targetSlots);
        QJsonObject definition;
        definition.insert(QStringLiteral("name"), name->text().trimmed());
        definition.insert(QStringLiteral("description"), description->text().trimmed());
        definition.insert(QStringLiteral("batch"), batchDefinition);
        const QByteArray idBytes = id->text().trimmed().toUtf8();
        const QByteArray versionBytes = version->text().trimmed().toUtf8();
        const QByteArray json = QJsonDocument(definition).toJson(QJsonDocument::Compact);
        char error[ErrorCapacity] = {};
        const bool saved = editing
            ? nn_app_update_dataset(application_.get(), idBytes.constData(), versionBytes.constData(),
                                    json.constData(), error, sizeof(error))
            : nn_app_create_dataset(application_.get(), idBytes.constData(), versionBytes.constData(),
                                    json.constData(), select->isChecked(), error, sizeof(error));
        if (!saved) {
            formError->setText(QString::fromUtf8(error));
            continue;
        }
        refreshAll();
        statusBar()->showMessage(editing ? tr("Dataset updated; current project saved")
                                         : tr("Dataset created; current project saved"), 6000);
        break;
    }
}
