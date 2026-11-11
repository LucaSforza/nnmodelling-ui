#include "MainWindow.hpp"
#include "MainWindowUtils.hpp"
#include "application/application.h"
#include <QCheckBox>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
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
#include <QTreeWidget>
#include <QTreeWidgetItem>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QMessageBox>
#include <QJsonParseError>
#include <QRegularExpression>

#include <cmath>

namespace {
bool readDatasetSourceFile(const QString &root, const QString &relativePath,
                           QByteArray *contents, bool *exists, QString *error)
{
    const QString clean = QDir::cleanPath(relativePath);
    if (QDir::isAbsolutePath(relativePath) || clean == QStringLiteral("..") ||
        clean.startsWith(QStringLiteral("../")) || clean == QStringLiteral(".")) {
        *error = QObject::tr("Dataset metadata contains an unsafe path: %1").arg(relativePath);
        return false;
    }
    QString current = QFileInfo(root).canonicalFilePath();
    for (const QString &part : clean.split(QLatin1Char('/'), Qt::SkipEmptyParts)) {
        current = QDir(current).filePath(part);
        const QFileInfo info(current);
        if (info.isSymLink()) {
            *error = QObject::tr("Dataset metadata may not traverse symbolic links.");
            return false;
        }
        if (!info.exists()) {
            *exists = false;
            if (contents) contents->clear();
            return true;
        }
    }
    const QFileInfo info(current);
    if (!info.isFile()) {
        *error = QObject::tr("Dataset metadata path is not a regular file: %1").arg(relativePath);
        return false;
    }
    if (!contents) {
        *exists = true;
        return true;
    }
    QFile file(current);
    if (!file.open(QIODevice::ReadOnly)) {
        *error = QObject::tr("Cannot read dataset metadata %1: %2").arg(relativePath, file.errorString());
        return false;
    }
    *contents = file.readAll();
    *exists = true;
    return true;
}

QJsonObject datasetSlots(const QJsonObject &definition, const QString &kind)
{
    return definition.value(QStringLiteral("batch")).toObject().value(kind).toObject();
}
}

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
    auto *importButton = new QPushButton(tr("Import dataset…"), &dialog);
    importButton->setObjectName(QStringLiteral("datasetManagerImport"));
    auto *editButton = new QPushButton(tr("Edit"), &dialog);
    editButton->setObjectName(QStringLiteral("datasetManagerEdit"));
    auto *selectButton = new QPushButton(tr("Activate"), &dialog);
    selectButton->setObjectName(QStringLiteral("datasetManagerActivate"));
    auto *deleteButton = new QPushButton(tr("Delete…"), &dialog);
    deleteButton->setObjectName(QStringLiteral("datasetManagerDelete"));
    auto *activeLabel = new QLabel(&dialog);
    activeLabel->setObjectName(QStringLiteral("datasetManagerActive"));
    layout->addWidget(activeLabel);
    auto *closeButton = new QPushButton(tr("Close"), &dialog);
    actions->addWidget(newButton);
    actions->addWidget(importButton);
    actions->addWidget(editButton);
    actions->addWidget(selectButton);
    actions->addWidget(deleteButton);
    actions->addStretch();
    actions->addWidget(closeButton);
    layout->addLayout(actions);

    auto refreshList = [this, list, editButton, selectButton, deleteButton, activeLabel](const QString &preferredIdentity) {
        list->clear();
        const NNProject *project = nn_app_project(application_.get());
        const NNDataset *active = project ? nn_project_active_dataset(project) : nullptr;
        activeLabel->setText(active
            ? tr("Active dataset: %1@%2").arg(QString::fromUtf8(active->id), QString::fromUtf8(active->version))
            : tr("Active dataset: none"));
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
        selectButton->setEnabled(list->currentItem() != nullptr);
        deleteButton->setEnabled(list->currentItem() != nullptr);
    };
    auto updateButtons = [list, editButton, selectButton, deleteButton] {
        const auto *item = list->currentItem();
        editButton->setEnabled(item != nullptr);
        selectButton->setEnabled(item != nullptr);
        deleteButton->setEnabled(item != nullptr);
    };
    QObject::connect(list, &QListWidget::currentItemChanged, &dialog,
                     [updateButtons] { updateButtons(); });
    QObject::connect(newButton, &QPushButton::clicked, &dialog, [this, &dialog, &refreshList] {
        createDataset();
        refreshList(QString());
        dialog.raise();
        dialog.activateWindow();
    });
    QObject::connect(importButton, &QPushButton::clicked, &dialog, [this, &dialog, &refreshList] {
        importDataset();
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
            QMessageBox::warning(this, tr("Dataset activation failed"), QString::fromUtf8(error));
            return;
        }
        refreshAll();
        refreshList(identity);
    });
    QObject::connect(deleteButton, &QPushButton::clicked, &dialog,
                     [this, list, &dialog, &refreshList] {
        const auto *item = list->currentItem();
        if (!item) return;
        const QByteArray id = item->data(Qt::UserRole).toString().toUtf8();
        const QByteArray version = item->data(Qt::UserRole + 1).toString().toUtf8();
        const QString identity = QStringLiteral("%1@%2").arg(QString::fromUtf8(id), QString::fromUtf8(version));
        const QString activeNotice = item->data(Qt::UserRole + 2).toBool()
            ? tr(" The active dataset will be cleared.") : QString();
        if (QMessageBox::question(&dialog, tr("Delete dataset"),
            tr("Delete %1 and its project files permanently? Current project edits will be saved.%2")
                .arg(identity, activeNotice), QMessageBox::Yes | QMessageBox::Cancel,
            QMessageBox::Cancel) != QMessageBox::Yes) return;
        char error[ErrorCapacity] = {};
        if (!nn_app_delete_dataset(application_.get(), id.constData(), version.constData(), error, sizeof(error))) {
            QMessageBox::warning(&dialog, tr("Dataset deletion failed"), QString::fromUtf8(error));
            return;
        }
        refreshAll();
        refreshList(QString());
        if (error[0]) QMessageBox::warning(&dialog, tr("Dataset files remain"), QString::fromUtf8(error));
        statusBar()->showMessage(tr("Dataset deleted; current project saved"), 6000);
    });
    QObject::connect(closeButton, &QPushButton::clicked, &dialog, &QDialog::reject);
    refreshList(QString());
    dialog.exec();
}

void MainWindow::importDataset() {
    if (!nn_app_project(application_.get())) {
        statusBar()->showMessage(tr("Open a project before importing a dataset"), 5000);
        return;
    }
    const QString source = QFileDialog::getExistingDirectory(
        this, tr("Choose dataset directory"), QString(), QFileDialog::ShowDirsOnly);
    if (source.isEmpty()) return;
    const QFileInfo sourceInfo(source);
    const QString root = sourceInfo.canonicalFilePath();
    if (root.isEmpty() || sourceInfo.isSymLink()) {
        QMessageBox::warning(this, tr("Dataset import failed"), tr("Choose a regular dataset directory, not a symbolic link."));
        return;
    }

    QByteArray manifestBytes;
    bool hasManifest = false;
    QString readError;
    if (!readDatasetSourceFile(root, QStringLiteral("manifest.json"), &manifestBytes,
                               &hasManifest, &readError)) {
        QMessageBox::warning(this, tr("Dataset import failed"), readError);
        return;
    }
    QJsonObject manifest;
    if (hasManifest) {
        QJsonParseError parseError{};
        const QJsonDocument parsed = QJsonDocument::fromJson(manifestBytes, &parseError);
        if (parseError.error != QJsonParseError::NoError || !parsed.isObject()) {
            QMessageBox::warning(this, tr("Dataset import failed"), tr("manifest.json is malformed."));
            return;
        }
        manifest = parsed.object();
        if ((manifest.contains(QStringLiteral("id")) && !manifest.value(QStringLiteral("id")).isString()) ||
            (manifest.contains(QStringLiteral("version")) && !manifest.value(QStringLiteral("version")).isString()) ||
            (manifest.contains(QStringLiteral("entrypoints")) && !manifest.value(QStringLiteral("entrypoints")).isObject())) {
            QMessageBox::warning(this, tr("Dataset import failed"), tr("manifest.json has invalid identity or entrypoint fields."));
            return;
        }
    }
    QJsonObject entrypoints = manifest.value(QStringLiteral("entrypoints")).toObject();
    if (entrypoints.contains(QStringLiteral("definition")) &&
        !entrypoints.value(QStringLiteral("definition")).isString()) {
        QMessageBox::warning(this, tr("Dataset import failed"), tr("The definition entrypoint must be a relative path string."));
        return;
    }
    QString definitionPath = entrypoints.value(QStringLiteral("definition")).toString();
    if (entrypoints.contains(QStringLiteral("definition")) && definitionPath.isEmpty()) {
        QMessageBox::warning(this, tr("Dataset import failed"), tr("The definition entrypoint cannot be empty."));
        return;
    }
    if (definitionPath.isEmpty()) definitionPath = QStringLiteral("dataset.json");
    QByteArray definitionBytes;
    bool hasDefinition = false;
    if (!readDatasetSourceFile(root, definitionPath, &definitionBytes, &hasDefinition, &readError)) {
        QMessageBox::warning(this, tr("Dataset import failed"), readError);
        return;
    }
    bool adapterDeclared = false;
    QString adapterPath = QStringLiteral("dataset.py");
    if (entrypoints.contains(QStringLiteral("python"))) {
        adapterDeclared = true;
        const QJsonValue pythonValue = entrypoints.value(QStringLiteral("python"));
        const QJsonObject python = pythonValue.toObject();
        if (!pythonValue.isObject() || !python.value(QStringLiteral("language")).isString() ||
            python.value(QStringLiteral("language")).toString() != QStringLiteral("python") ||
            !python.value(QStringLiteral("file")).isString() || python.value(QStringLiteral("file")).toString().isEmpty()) {
            QMessageBox::warning(this, tr("Dataset import failed"), tr("The Python adapter entrypoint is malformed."));
            return;
        }
        adapterPath = python.value(QStringLiteral("file")).toString();
    }
    bool hasAdapter = false;
    if (!readDatasetSourceFile(root, adapterPath, nullptr, &hasAdapter, &readError)) {
        QMessageBox::warning(this, tr("Dataset import failed"), readError);
        return;
    }
    QJsonObject definition;
    if (hasDefinition) {
        QJsonParseError parseError{};
        const QJsonDocument parsed = QJsonDocument::fromJson(definitionBytes, &parseError);
        if (parseError.error != QJsonParseError::NoError || !parsed.isObject()) {
            QMessageBox::warning(this, tr("Dataset import failed"), tr("The dataset definition is malformed."));
            return;
        }
        definition = parsed.object();
        const QJsonValue batchValue = definition.value(QStringLiteral("batch"));
        if ((definition.contains(QStringLiteral("name")) && !definition.value(QStringLiteral("name")).isString()) ||
            (definition.contains(QStringLiteral("description")) && !definition.value(QStringLiteral("description")).isString()) ||
            (definition.contains(QStringLiteral("batch")) && !batchValue.isObject())) {
            QMessageBox::warning(this, tr("Dataset import failed"), tr("The dataset definition has invalid name, description, or batch fields."));
            return;
        }
        const QJsonObject batchObject = batchValue.toObject();
        for (const QString &kind : {QStringLiteral("inputs"), QStringLiteral("targets")}) {
            const QJsonValue slotsValue = batchObject.value(kind);
            if ((batchObject.contains(kind) && !slotsValue.isObject())) {
                QMessageBox::warning(this, tr("Dataset import failed"), tr("The dataset definition has invalid %1 slots.").arg(kind));
                return;
            }
            const QJsonObject slotsObject = slotsValue.toObject();
            for (auto it = slotsObject.begin(); it != slotsObject.end(); ++it) {
                const QJsonObject tensor = it.value().toObject();
                if (!it.value().isObject() ||
                    (tensor.contains(QStringLiteral("dtype")) && !tensor.value(QStringLiteral("dtype")).isString()) ||
                    (tensor.contains(QStringLiteral("shape")) && !tensor.value(QStringLiteral("shape")).isArray())) {
                    QMessageBox::warning(this, tr("Dataset import failed"), tr("The dataset definition has malformed tensor slot metadata."));
                    return;
                }
                static const QSet<QString> dtypes = {"float16", "bfloat16", "float32", "float64", "int8", "int16", "int32", "int64", "uint8", "bool"};
                if (it.key().isEmpty() ||
                    (tensor.contains(QStringLiteral("dtype")) && !dtypes.contains(tensor.value(QStringLiteral("dtype")).toString()))) {
                    QMessageBox::warning(this, tr("Dataset import failed"), tr("The dataset definition has invalid tensor slot metadata."));
                    return;
                }
                if (!tensor.contains(QStringLiteral("shape"))) continue;
                const QJsonArray shape = tensor.value(QStringLiteral("shape")).toArray();
                if (shape.isEmpty() || shape.size() > 64) {
                    QMessageBox::warning(this, tr("Dataset import failed"), tr("The dataset definition has invalid tensor slot metadata."));
                    return;
                }
                for (const QJsonValue &dimension : shape) {
                    const double numeric = dimension.toDouble();
                    if (!(dimension.isString() && dimension.toString() == QStringLiteral("B")) &&
                        !(dimension.isDouble() && numeric > 0 && std::floor(numeric) == numeric)) {
                        QMessageBox::warning(this, tr("Dataset import failed"), tr("The dataset definition has malformed tensor dimensions."));
                        return;
                    }
                }
            }
        }
    }

    QDialog dialog(this);
    dialog.setObjectName(QStringLiteral("datasetImportDialog"));
    dialog.setWindowTitle(tr("Import dataset — saves project"));
    dialog.resize(760, 620);
    auto *layout = new QVBoxLayout(&dialog);
    auto *form = new QFormLayout;
    auto *id = new QLineEdit(manifest.value(QStringLiteral("id")).toString(), &dialog);
    id->setObjectName(QStringLiteral("datasetImportId"));
    if (id->text().isEmpty()) id->setText(QFileInfo(root).fileName().replace(QRegularExpression(QStringLiteral("[^A-Za-z0-9._-]")), QStringLiteral("-")));
    auto *version = new QLineEdit(manifest.value(QStringLiteral("version")).toString(QStringLiteral("1.0.0")), &dialog);
    version->setObjectName(QStringLiteral("datasetImportVersion"));
    auto *name = new QLineEdit(definition.value(QStringLiteral("name")).toString(), &dialog);
    name->setObjectName(QStringLiteral("datasetImportName"));
    auto *description = new QLineEdit(definition.value(QStringLiteral("description")).toString(), &dialog);
    description->setObjectName(QStringLiteral("datasetImportDescription"));
    form->addRow(tr("ID"), id);
    form->addRow(tr("Version"), version);
    form->addRow(tr("Name"), name);
    form->addRow(tr("Description"), description);
    layout->addLayout(form);
    auto *notice = new QLabel(&dialog);
    notice->setWordWrap(true);
    notice->setObjectName(QStringLiteral("datasetImportNotice"));
    notice->setText(hasAdapter
        ? tr("Source: %1. Existing Python adapter %2 will be retained and imported.").arg(root, adapterPath)
        : adapterDeclared
            ? tr("Source: %1. The declared Python adapter %2 is missing. Import creates a nonfunctional scaffold; this dataset will not be trainable until its adapter is implemented.").arg(root, adapterPath)
            : tr("Source: %1. No Python adapter was found. Import creates a nonfunctional scaffold; this dataset will not be trainable until its adapter is implemented.").arg(root));
    layout->addWidget(notice);

    auto addSlotTable = [&dialog, layout](const QString &title, const QString &objectName, const QJsonObject &initial) {
        auto *heading = new QLabel(title + QObject::tr(" — dimensions use commas; B marks the batch dimension."), &dialog);
        heading->setWordWrap(true);
        layout->addWidget(heading);
        auto *table = new QTableWidget(0, 3, &dialog);
        table->setObjectName(objectName);
        table->setMinimumHeight(110);
        table->setHorizontalHeaderLabels({QObject::tr("Slot name"), QObject::tr("Dtype"), QObject::tr("Shape")});
        table->horizontalHeader()->setStretchLastSection(true);
        table->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
        for (auto it = initial.begin(); it != initial.end(); ++it) {
            const QJsonObject tensor = it.value().toObject();
            QStringList dims;
            for (const QJsonValue &dim : tensor.value(QStringLiteral("shape")).toArray())
                dims.append(dim.isString() ? dim.toString() : QString::number(dim.toInteger()));
            addTableRow(table, {it.key(), tensor.value(QStringLiteral("dtype")).toString(), dims.join(QStringLiteral(", "))});
        }
        layout->addWidget(table);
        auto *row = new QHBoxLayout;
        auto *add = new QPushButton(QObject::tr("Add row"), &dialog);
        auto *remove = new QPushButton(QObject::tr("Remove row"), &dialog);
        row->addWidget(add); row->addWidget(remove); row->addStretch(); layout->addLayout(row);
        QObject::connect(add, &QPushButton::clicked, table, [table] { addTableRow(table, {QString(), QStringLiteral("float32"), QStringLiteral("B")}); });
        QObject::connect(remove, &QPushButton::clicked, table, [table] { if (table->currentRow() >= 0) table->removeRow(table->currentRow()); });
        return table;
    };
    const QJsonObject batch = definition.value(QStringLiteral("batch")).toObject();
    auto *inputs = addSlotTable(tr("Input slots"), QStringLiteral("datasetImportInputs"), batch.value(QStringLiteral("inputs")).toObject());
    auto *targets = addSlotTable(tr("Target slots"), QStringLiteral("datasetImportTargets"), batch.value(QStringLiteral("targets")).toObject());
    auto *errorLabel = new QLabel(&dialog);
    errorLabel->setObjectName(QStringLiteral("datasetImportError"));
    errorLabel->setWordWrap(true);
    errorLabel->setStyleSheet(QStringLiteral("color: #a12b2b;"));
    auto *buttons = new QDialogButtonBox(QDialogButtonBox::Save | QDialogButtonBox::Cancel, &dialog);
    buttons->button(QDialogButtonBox::Save)->setText(tr("Import and save project"));
    QObject::connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    QObject::connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    layout->addWidget(errorLabel);
    layout->addWidget(buttons);

    while (dialog.exec() == QDialog::Accepted) {
        auto collectSlots = [](QTableWidget *table, const QJsonObject &original, QJsonObject *slotMap) {
            static const QSet<QString> dtypes = {"float16", "bfloat16", "float32", "float64", "int8", "int16", "int32", "int64", "uint8", "bool"};
            QSet<QString> seen;
            for (int r = 0; r < table->rowCount(); ++r) {
                const QString key = cellText(table, r, 0).trimmed();
                const QString dtype = cellText(table, r, 1).trimmed();
                if (key.isEmpty() || !dtypes.contains(dtype)) return false;
                QJsonArray shape;
                const QStringList parts = cellText(table, r, 2).split(',', Qt::KeepEmptyParts);
                if (parts.isEmpty() || parts.size() > 64) return false;
                for (QString part : parts) {
                    part = part.trimmed();
                    if (part == QStringLiteral("B")) shape.append(part);
                    else { bool ok = false; const qlonglong value = part.toLongLong(&ok); if (!ok || value <= 0) return false; shape.append(QJsonValue(qint64(value))); }
                }
                if (seen.contains(key)) return false;
                seen.insert(key);
                QJsonObject tensor = original.value(key).toObject();
                tensor.insert(QStringLiteral("dtype"), dtype);
                tensor.insert(QStringLiteral("shape"), shape);
                slotMap->insert(key, tensor);
            }
            return true;
        };
        QJsonObject inputSlots, targetSlots;
        if (!collectSlots(inputs, datasetSlots(definition, QStringLiteral("inputs")), &inputSlots) ||
            inputSlots.isEmpty() ||
            !collectSlots(targets, datasetSlots(definition, QStringLiteral("targets")), &targetSlots)) {
            errorLabel->setText(tr("Input slots are required; use unique names, supported dtypes and positive dimensions (or B)."));
            continue;
        }
        manifest.insert(QStringLiteral("schemaVersion"), manifest.value(QStringLiteral("schemaVersion")).isUndefined() ? 1 : manifest.value(QStringLiteral("schemaVersion")));
        manifest.insert(QStringLiteral("id"), id->text().trimmed());
        manifest.insert(QStringLiteral("version"), version->text().trimmed());
        entrypoints.insert(QStringLiteral("definition"), definitionPath);
        manifest.insert(QStringLiteral("entrypoints"), entrypoints);
        QJsonObject batchOut = definition.value(QStringLiteral("batch")).toObject();
        batchOut.insert(QStringLiteral("inputs"), inputSlots);
        batchOut.insert(QStringLiteral("targets"), targetSlots);
        definition.insert(QStringLiteral("name"), name->text().trimmed());
        definition.insert(QStringLiteral("description"), description->text().trimmed());
        definition.insert(QStringLiteral("batch"), batchOut);
        const QByteArray manifestJson = QJsonDocument(manifest).toJson(QJsonDocument::Compact);
        const QByteArray definitionJson = QJsonDocument(definition).toJson(QJsonDocument::Compact);
        const QByteArray sourceBytes = root.toUtf8();
        char error[ErrorCapacity] = {};
        if (!nn_app_import_dataset(application_.get(), sourceBytes.constData(), manifestJson.constData(),
                                   definitionJson.constData(), error, sizeof(error))) {
            errorLabel->setText(QString::fromUtf8(error));
            continue;
        }
        refreshAll();
        statusBar()->showMessage(tr("Dataset imported; current project saved"), 6000);
        break;
    }
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
        auto *heading = new QLabel(title + QObject::tr(
            " — Enter dimensions separated by commas. B is the batch dimension: B,4 means four values per sample; B,1 means one. Dtypes: float16, bfloat16, float32/64, int8/16/32/64, uint8, bool."), content);
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
        auto buildSlots = [](QTableWidget *table, QJsonObject *out) -> bool {
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
