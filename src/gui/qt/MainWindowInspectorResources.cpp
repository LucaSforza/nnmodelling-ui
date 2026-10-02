#include "MainWindow.hpp"
#include "GraphScene.hpp"
#include "MainWindowUtils.hpp"
#include "application/application.h"
#include "catalog/catalog.h"
#include "inference/inference.h"
#include "model/model.h"
#include "project/project.h"
#include <QBrush>
#include <QCheckBox>
#include <QComboBox>
#include <QLineEdit>
#include <QLabel>
#include <QMessageBox>
#include <QPushButton>
#include <QStatusBar>
#include <QTimer>
#include <QTreeWidget>
#include <QTreeWidgetItem>
#include <QStringList>
#include <map>
#include <cstring>
#include <string>

using namespace MainWindowUtils;

void MainWindow::refreshInspector() {
    if (!inspector_ || refreshing_) return;
    refreshing_ = true;
    inspector_->clear();
    const QString selectedId = scene_ ? scene_->selectedNodeId() : QString();
    const NNModel *model = nn_app_model(application_.get());
    if (selectedId.isEmpty() || !model) {
        refreshing_ = false;
        return;
    }
    const QByteArray idBytes = selectedId.toUtf8();
    const NNNode *node = nn_model_find_node(model, idBytes.constData());
    if (!node) {
        refreshing_ = false;
        return;
    }
    const QString nodeId = QString::fromUtf8(node->id);
    auto *nameRow = new QTreeWidgetItem(inspector_, {tr("Name"), QString::fromUtf8(node->label)});
    auto *nameEdit = new QLineEdit(QString::fromUtf8(node->label), inspector_);
    inspector_->setItemWidget(nameRow, 1, nameEdit);
    connect(nameEdit, &QLineEdit::editingFinished, this, [this, nameEdit, nodeId] {
        editNodeName(nodeId, nameEdit->text());
    });

    const NNProject *project = nn_app_project(application_.get());
    const NNPackage *package = project
        ? nn_catalog_find(nn_project_catalog(project), node->package_id, node->package_version)
        : nullptr;
    auto *packageRow = new QTreeWidgetItem(inspector_, {tr("Package"),
        QStringLiteral("%1@%2").arg(QString::fromUtf8(node->package_id),
                                     QString::fromUtf8(node->package_version))});
    packageRow->setToolTip(1, package && package->description
        ? QString::fromUtf8(package->description) : QString());
    if (package) {
        char analysisError[ErrorCapacity] = {};
        const NNInferenceReport *analysis = nn_app_analysis(application_.get(), analysisError,
                                                            sizeof(analysisError));
        for (size_t i = 0; analysis && i < nn_inference_count(analysis); ++i) {
            const NNInferenceResult *result = nn_inference_at(analysis, i);
            if (!result || result->status != NN_INFERENCE_SUCCESS ||
                std::strcmp(result->node_id ? result->node_id : "", node->id) != 0) continue;
            auto addTensor = [this](QTreeWidgetItem *parent, const QString &label,
                                    const char *dtype, const char *const *dimensions,
                                    size_t dimensionCount) {
                QStringList shape;
                for (size_t j = 0; j < dimensionCount; ++j)
                    shape.push_back(QString::fromUtf8(dimensions && dimensions[j] ? dimensions[j] : "?"));
                const QString tensor = QStringLiteral("%1[%2]").arg(
                    QString::fromUtf8(dtype ? dtype : ""), shape.join(QStringLiteral(", ")));
                new QTreeWidgetItem(parent, {label, tensor});
            };
            if (result->output_count) {
                auto *outputsRow = new QTreeWidgetItem(inspector_, {tr("Successful outputs"), QString()});
                for (size_t j = 0; j < result->output_count; ++j) {
                    const NNInferenceTensor &tensor = result->outputs[j];
                    const QString label = QStringLiteral("%1 (%2)").arg(
                        QString::fromUtf8(tensor.handle_id ? tensor.handle_id : ""),
                        QString::fromUtf8(tensor.type ? tensor.type : ""));
                    addTensor(outputsRow, label, tensor.dtype, tensor.dimensions,
                              tensor.dimension_count);
                }
            } else if (result->dtype) {
                auto *consumed = new QTreeWidgetItem(inspector_, {tr("Consumed tensor"), QString()});
                addTensor(consumed, tr("Tensor"), result->dtype, result->dimensions,
                          result->dimension_count);
            }
            break;
        }
        const QString kind = package->kind ? QString::fromUtf8(package->kind) : QString();
        if ((kind == QStringLiteral("output") || kind == QStringLiteral("loss-output")) &&
            node->scope_id && *node->scope_id) {
            const NNNode *owner = nn_model_find_node(model, node->scope_id);
            const NNPackage *ownerPackage = owner
                ? nn_catalog_find(nn_project_catalog(project), owner->package_id, owner->package_version)
                : nullptr;
            auto *mappingRow = new QTreeWidgetItem(inspector_, {tr("Boundary mapping"), QString()});
            auto *mapping = new QComboBox(inspector_);
            mapping->setObjectName(QStringLiteral("boundaryMapping"));
            const QString currentMapping = node->boundary_handle_id
                ? QString::fromUtf8(node->boundary_handle_id) : QString();
            if (currentMapping.isEmpty()) mapping->addItem(tr("(unmapped)"), QString());
            const QString expectedType = kind == QStringLiteral("output")
                ? QStringLiteral("output") : QStringLiteral("loss");
            for (size_t i = 0; ownerPackage && i < ownerPackage->output_count; ++i) {
                const NNOutputDef &output = ownerPackage->outputs[i];
                if (!output.id || !output.type || expectedType != QString::fromUtf8(output.type)) continue;
                mapping->addItem(QStringLiteral("%1 (%2)").arg(QString::fromUtf8(output.id),
                                                                 QString::fromUtf8(output.type)),
                                 QString::fromUtf8(output.id));
            }
            int mappingIndex = mapping->findData(currentMapping);
            if (mappingIndex < 0 && !currentMapping.isEmpty()) {
                mapping->addItem(tr("Invalid mapping: %1").arg(currentMapping), currentMapping);
                mappingIndex = mapping->count() - 1;
            }
            mapping->setCurrentIndex(mappingIndex);
            inspector_->setItemWidget(mappingRow, 1, mapping);
            connect(mapping, &QComboBox::currentIndexChanged, this,
                    [this, mapping, nodeId](int index) {
                const QByteArray id = nodeId.toUtf8();
                const QByteArray handle = mapping->itemData(index).toString().toUtf8();
                char error[ErrorCapacity] = {};
                if (!nn_app_set_boundary_handle(application_.get(), id.constData(), handle.constData(),
                                                error, sizeof(error))) {
                    QMessageBox::warning(this, tr("Boundary mapping rejected"), QString::fromUtf8(error));
                }
                QMetaObject::invokeMethod(this, [this] { refreshAll(); }, Qt::QueuedConnection);
            });
        }
        auto *parameters = new QTreeWidgetItem(inspector_, {tr("Parameters"), QString()});
        parameters->setExpanded(true);
        for (size_t i = 0; i < package->parameter_count; ++i) {
            const NNParameterDef &definition = package->parameters[i];
            if (!definition.key) continue;
            const QString key = QString::fromUtf8(definition.key);
            char *valueText = nn_app_parameter_text(application_.get(), node->id, definition.key);
            const QString value = valueText ? QString::fromUtf8(valueText) : QString();
            nn_app_free_text(valueText);
            auto *row = new QTreeWidgetItem(parameters, {key, value});
            QWidget *editor = nullptr;
            const QString type = definition.type ? QString::fromUtf8(definition.type) : QString();
            if (definition.choice_count > 0) {
                auto *combo = new QComboBox(inspector_);
                for (size_t j = 0; j < definition.choice_count; ++j) {
                    const QString choice = QString::fromUtf8(definition.choices[j]);
                    combo->addItem(choice);
                }
                combo->setCurrentText(value);
                connect(combo, &QComboBox::currentTextChanged, this,
                        [this, key, nodeId](const QString &next) {
                    editNodeParameter(nodeId, key, next);
                });
                editor = combo;
            } else if (type == QStringLiteral("stereotype")) {
                auto *unsupported = new QLabel(tr("Unsupported native value"), inspector_);
                unsupported->setToolTip(value);
                editor = unsupported;
            } else if (type == QStringLiteral("boolean") || type == QStringLiteral("bool")) {
                auto *check = new QCheckBox(inspector_);
                check->setChecked(value == QStringLiteral("true"));
                connect(check, &QCheckBox::toggled, this, [this, key, nodeId](bool checked) {
                    editNodeParameter(nodeId, key, checked ? QStringLiteral("true")
                                                           : QStringLiteral("false"));
                });
                editor = check;
            } else if (type == QStringLiteral("integer")) {
                auto *line = new QLineEdit(value, inspector_);
                connect(line, &QLineEdit::editingFinished, this, [this, key, nodeId, line] {
                    editNodeParameter(nodeId, key, line->text());
                });
                editor = line;
            } else if (type == QStringLiteral("number") || type == QStringLiteral("real")) {
                auto *line = new QLineEdit(value, inspector_);
                connect(line, &QLineEdit::editingFinished, this, [this, key, nodeId, line] {
                    editNodeParameter(nodeId, key, line->text());
                });
                editor = line;
            } else {
                auto *line = new QLineEdit(value, inspector_);
                connect(line, &QLineEdit::editingFinished, this, [this, key, nodeId, line] {
                    editNodeParameter(nodeId, key, line->text());
                });
                editor = line;
            }
            inspector_->setItemWidget(row, 1, editor);
        }
    }
    inspector_->expandAll();
    refreshing_ = false;
}

void MainWindow::refreshResources() {
    resources_->clear();
    const NNProject *project = nn_app_project(application_.get());
    if (!project) return;
    auto *datasets = new QTreeWidgetItem(resources_, {tr("Datasets")});
    datasets->setExpanded(true);
    for (size_t i = 0; i < nn_project_dataset_count(project); ++i) {
        const NNDataset *dataset = nn_project_dataset_at(project, i);
        if (!dataset) continue;
        const QString identity = QStringLiteral("%1@%2").arg(
            QString::fromUtf8(dataset->id), QString::fromUtf8(dataset->version));
        auto *datasetItem = new QTreeWidgetItem(datasets,
            {QStringLiteral("%1  %2").arg(textOr(QString::fromUtf8(dataset->name), dataset->id), identity)});
        datasetItem->setData(0, IdRole, identity);
        datasetItem->setData(0, PackageVersionRole, QString::fromUtf8(dataset->id));
        datasetItem->setData(0, DatasetVersionRole, QString::fromUtf8(dataset->version));
        if (nn_project_active_dataset(project) == dataset)
            datasetItem->setText(0, QStringLiteral("✓  %1").arg(datasetItem->text(0)));
        datasetItem->setToolTip(0, dataset->path ? QString::fromUtf8(dataset->path) : QString());
        for (size_t j = 0; j < dataset->input_count; ++j) {
            const NNTensorSlot &slot = dataset->inputs[j];
            new QTreeWidgetItem(datasetItem, {QStringLiteral("Input: %1 [%2]").arg(
                QString::fromUtf8(slot.name), QString::fromUtf8(slot.dtype))});
        }
        for (size_t j = 0; j < dataset->target_count; ++j) {
            const NNTensorSlot &slot = dataset->targets[j];
            new QTreeWidgetItem(datasetItem, {QStringLiteral("Target: %1 [%2]").arg(
                QString::fromUtf8(slot.name), QString::fromUtf8(slot.dtype))});
        }
    }
    auto *packages = new QTreeWidgetItem(resources_, {tr("Packages")});
    packages->setExpanded(true);
    const NNCatalog *catalog = nn_project_catalog(project);
    for (size_t i = 0; catalog && i < nn_catalog_count(catalog); ++i) {
        const NNPackage *package = nn_catalog_at(catalog, i);
        if (!package) continue;
        auto *item = new QTreeWidgetItem(packages, {QStringLiteral("%1@%2").arg(
            package->id ? QString::fromUtf8(package->id) : QString(),
            package->version ? QString::fromUtf8(package->version) : QString())});
        item->setForeground(0, QBrush(readablePackageColor(package)));
        for (size_t j = 0; j < package->dependency_count; ++j) {
            const NNPackageDependency &dependency = package->dependencies[j];
            new QTreeWidgetItem(item, {QStringLiteral("Requires %1 %2").arg(
                QString::fromUtf8(dependency.id), QString::fromUtf8(dependency.version_constraint))});
        }
    }
    resources_->expandAll();
}

void MainWindow::updateWindowTitle() {
    const NNProject *project = nn_app_project(application_.get());
    if (!project) {
        setWindowTitle(tr("NNModelling"));
        return;
    }
    const QString marker = nn_project_dirty(project) ? QStringLiteral(" *") : QString();
    setWindowTitle(QStringLiteral("%1%2 — NNModelling").arg(
        QString::fromUtf8(nn_project_name(project)), marker));
}

void MainWindow::editNodeName(const QString &nodeId, const QString &name) {
    if (refreshing_ || !scene_) return;
    const QByteArray id = nodeId.toUtf8();
    const QByteArray label = name.trimmed().toUtf8();
    char error[ErrorCapacity] = {};
    if (!nn_app_rename_node(application_.get(), id.constData(), label.constData(),
                            error, sizeof(error))) {
        QMessageBox::warning(this, tr("Rename failed"), QString::fromUtf8(error));
    }
    QTimer::singleShot(0, this, [this] { refreshAll(); });
}

void MainWindow::editNodeParameter(const QString &nodeId, const QString &key,
                                   const QString &value) {
    if (refreshing_ || !scene_) return;
    const QByteArray id = nodeId.toUtf8();
    const QByteArray keyBytes = key.toUtf8();
    const QByteArray text = value.toUtf8();
    char error[ErrorCapacity] = {};
    if (!nn_app_set_parameter_text(application_.get(), id.constData(), keyBytes.constData(),
                                   text.constData(), error, sizeof(error))) {
        QMessageBox::warning(this, tr("Parameter rejected"), QString::fromUtf8(error));
        return;
    }
    QTimer::singleShot(0, this, [this] { refreshAll(); });
}

void MainWindow::selectResource(QTreeWidgetItem *item, int) {
    if (!item || item->data(0, IdRole).toString().isEmpty()) return;
    // Resource identities are shown here; selection never changes project ownership.
    statusBar()->showMessage(item->toolTip(0), 5000);
}

void MainWindow::selectDataset(QTreeWidgetItem *item, int column) {
    if (!item) return;
    if (item->data(0, PackageVersionRole).toString().isEmpty()) {
        selectResource(item, column);
        return;
    }
    const QByteArray id = item->data(0, PackageVersionRole).toString().toUtf8();
    const QByteArray version = item->data(0, DatasetVersionRole).toString().toUtf8();
    char error[ErrorCapacity] = {};
    if (!nn_app_select_dataset(application_.get(), id.constData(), version.constData(), error, sizeof(error))) {
        QMessageBox::warning(this, tr("Dataset selection failed"), QString::fromUtf8(error));
        return;
    }
    refreshAll();
    statusBar()->showMessage(tr("Selected dataset %1@%2").arg(QString::fromUtf8(id), QString::fromUtf8(version)), 5000);
}
