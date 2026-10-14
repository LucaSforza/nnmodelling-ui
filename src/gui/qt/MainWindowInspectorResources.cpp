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
#include <QHBoxLayout>
#include <QLineEdit>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QLabel>
#include <QMessageBox>
#include <QPushButton>
#include <QRegularExpression>
#include <QStatusBar>
#include <QTimer>
#include <QTreeWidget>
#include <QTreeWidgetItem>

namespace {
QJsonValue parameterEditorValue(const QString &type, const QString &text)
{
    if (text.startsWith(QStringLiteral("$$"))) return text.mid(1);
    static const QRegularExpression referencePattern(QStringLiteral("^\\$([A-Za-z_][A-Za-z0-9_-]*)$"));
    const auto reference = referencePattern.match(text);
    if (reference.hasMatch()) return QJsonObject{{QStringLiteral("$ref"), reference.captured(1)}};
    if (text.startsWith(QStringLiteral("$(")) && text.endsWith(QLatin1Char(')')))
        return QJsonObject{{QStringLiteral("$lua"), text.mid(2, text.size() - 3)}};
    if (text.startsWith(QLatin1Char('$'))) return {};
    if (type == QStringLiteral("string") || type == QStringLiteral("dtype")) return text;
    QJsonParseError error;
    const QJsonDocument parsed = QJsonDocument::fromJson(text.toUtf8(), &error);
    if (type == QStringLiteral("json"))
        return error.error == QJsonParseError::NoError && parsed.isArray() ? QJsonValue(parsed.array()) : QJsonValue();
    const QJsonDocument scalar = QJsonDocument::fromJson(QByteArray("[") + text.toUtf8() + QByteArray("]"), &error);
    return error.error == QJsonParseError::NoError && scalar.isArray() && scalar.array().size() == 1
        ? scalar.array().at(0) : QJsonValue();
}

QString parameterBindingText(const QJsonValue &value)
{
    if (!value.isObject()) return {};
    const QJsonObject object = value.toObject();
    if (object.size() == 1 && object.value(QStringLiteral("$ref")).isString())
        return QLatin1Char('$') + object.value(QStringLiteral("$ref")).toString();
    if (object.size() == 1 && object.value(QStringLiteral("$lua")).isString())
        return QStringLiteral("$(%1)").arg(object.value(QStringLiteral("$lua")).toString());
    return {};
}

QString parameterEditorText(const QString &raw, const QString &type)
{
    const QJsonDocument document = QJsonDocument::fromJson(raw.toUtf8());
    if (document.isObject()) {
        const QString binding = parameterBindingText(document.object());
        if (!binding.isEmpty()) return binding;
    }
    return (type == QStringLiteral("string") || type == QStringLiteral("dtype")) &&
           raw.startsWith(QLatin1Char('$')) ? QLatin1Char('$') + raw : raw;
}
}
#include <QStringList>
#include <map>
#include <algorithm>
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
            const QString rawValue = valueText ? QString::fromUtf8(valueText) : QString();
            nn_app_free_text(valueText);
            const QString type = definition.type ? QString::fromUtf8(definition.type) : QString();
            const QString value = parameterEditorText(rawValue, type);
            char previewError[ErrorCapacity] = {};
            char *previewText = nn_app_resolved_parameter_text(application_.get(), node->id,
                definition.key, previewError, sizeof(previewError));
            const QString resolvedValue = previewText ? QString::fromUtf8(previewText) : QString();
            nn_app_free_text(previewText);
            const bool hasBinding =
                !parameterBindingText(QJsonDocument::fromJson(rawValue.toUtf8()).object()).isEmpty();
            const QString displayValue = hasBinding && !resolvedValue.isEmpty() ? resolvedValue : value;
            auto *row = new QTreeWidgetItem(parameters, {key, value});
            QWidget *editor = nullptr;
            if (definition.choice_count > 0) {
                auto *combo = new QComboBox(inspector_);
                for (size_t j = 0; j < definition.choice_count; ++j) {
                    const QString choice = QString::fromUtf8(definition.choices[j]);
                    combo->addItem(choice);
                }
                combo->setCurrentText(displayValue);
                connect(combo, &QComboBox::currentTextChanged, this,
                        [this, key, nodeId](const QString &next) {
                    editNodeParameter(nodeId, key, next);
                });
                editor = combo;
            } else if (type == QStringLiteral("stereotype")) {
                auto *combo = new QComboBox(inspector_);
                combo->setObjectName(QStringLiteral("stereotypeReference_%1").arg(key));
                QJsonObject reference = QJsonDocument::fromJson(displayValue.toUtf8()).object();
                const QJsonObject authoredReference = QJsonDocument::fromJson(rawValue.toUtf8()).object();
                const QString currentId = reference.value(QStringLiteral("id")).toString();
                const QString currentVersion = reference.value(QStringLiteral("version")).toString();
                combo->addItem(tr("(choose package)"), QJsonObject());
                const NNCatalog *catalog = nn_project_catalog(project);
                int currentIndex = 0;
                for (size_t j = 0; j < nn_catalog_count(catalog); ++j) {
                    const NNPackage *candidate = nn_catalog_at(catalog, j);
                    if (!candidate->kind || std::strcmp(candidate->kind, "subflow") == 0 ||
                        (definition.kind && std::strcmp(candidate->kind, definition.kind) != 0)) continue;
                    QJsonObject defaults;
                    for (size_t p = 0; p < candidate->parameter_count; ++p) {
                        const NNParameterDef &def = candidate->parameters[p];
                        QJsonValue fallback;
                        if (def.has_default) {
                            switch (def.default_value.type) {
                            case NN_PARAMETER_BOOLEAN: fallback = def.default_value.as.boolean; break;
                            case NN_PARAMETER_INTEGER: fallback = double(def.default_value.as.integer); break;
                            case NN_PARAMETER_NUMBER: fallback = def.default_value.as.number; break;
                            case NN_PARAMETER_STRING: fallback = QString::fromUtf8(def.default_value.as.string); break;
                            case NN_PARAMETER_JSON: {
                                const QJsonDocument json = QJsonDocument::fromJson(def.default_value.as.string);
                                fallback = json.isArray() ? QJsonValue(json.array()) : QJsonValue(json.object());
                                break;
                            }
                            default: break;
                            }
                        } else if (std::strcmp(def.type, "integer") == 0)
                            fallback = def.has_minimum ? std::max(1.0, def.minimum) : 1.0;
                        else if (std::strcmp(def.type, "number") == 0)
                            fallback = def.has_minimum ? def.minimum : 0.0;
                        else if (std::strcmp(def.type, "boolean") == 0) fallback = false;
                        else if (def.choice_count) fallback = QString::fromUtf8(def.choices[0]);
                        else if (std::strcmp(def.type, "string") == 0) fallback = QString();
                        defaults.insert(QString::fromUtf8(def.key), fallback);
                    }
                    QJsonObject next{{QStringLiteral("id"), QString::fromUtf8(candidate->id)},
                                     {QStringLiteral("version"), QString::fromUtf8(candidate->version)},
                                     {QStringLiteral("parameters"), defaults}};
                    combo->addItem(QStringLiteral("%1 (%2@%3)").arg(QString::fromUtf8(candidate->name),
                        QString::fromUtf8(candidate->id), QString::fromUtf8(candidate->version)), next);
                    if (currentId == QString::fromUtf8(candidate->id) &&
                        nn_catalog_resolve(catalog, candidate->id, currentVersion.toUtf8().constData()) == candidate)
                        currentIndex = combo->count() - 1;
                }
                combo->setCurrentIndex(currentIndex);
                connect(combo, &QComboBox::currentIndexChanged, this, [this, combo, key, nodeId](int index) {
                    if (index > 0) editNodeParameter(nodeId, key, QString::fromUtf8(
                        QJsonDocument(combo->itemData(index).toJsonObject()).toJson(QJsonDocument::Compact)));
                });
                const NNPackage *referenced = currentIndex > 0
                    ? nn_catalog_resolve(catalog, currentId.toUtf8().constData(),
                                         currentVersion.toUtf8().constData()) : nullptr;
                const QJsonObject committed = hasBinding
                    ? reference.value(QStringLiteral("parameters")).toObject()
                    : authoredReference.value(QStringLiteral("parameters")).toObject();
                const QJsonObject resolvedParameters = reference.value(QStringLiteral("parameters")).toObject();
                const QJsonObject defaults = combo->currentData().toJsonObject()
                    .value(QStringLiteral("parameters")).toObject();
                for (size_t p = 0; referenced && p < referenced->parameter_count; ++p) {
                    const NNParameterDef &def = referenced->parameters[p];
                    const QString parameterKey = QString::fromUtf8(def.key);
                    const QJsonValue authored = committed.contains(parameterKey)
                        ? committed.value(parameterKey) : defaults.value(parameterKey);
                    const QJsonValue resolved = resolvedParameters.contains(parameterKey)
                        ? resolvedParameters.value(parameterKey) : authored;
                    const QString nestedBinding = parameterBindingText(authored);
                    const QJsonValue current = nestedBinding.isEmpty() ? authored : resolved;
                    auto *fieldRow = new QTreeWidgetItem(row, {parameterKey, QString()});
                    // Read siblings at submission: several fields may commit before queued refresh.
                    auto submit = [this, nodeId, key, parameterKey, reference](const QJsonValue &next) {
                        const QByteArray id = nodeId.toUtf8(), parameter = key.toUtf8();
                        char *text = nn_app_parameter_text(application_.get(), id.constData(), parameter.constData());
                        QJsonObject updated = QJsonDocument::fromJson(text ? QByteArray(text) : QByteArray()).object();
                        nn_app_free_text(text);
                        if (updated.isEmpty()) updated = reference;
                        QJsonObject values = updated.value(QStringLiteral("parameters")).toObject();
                        values.insert(parameterKey, next);
                        updated.insert(QStringLiteral("parameters"), values);
                        editNodeParameter(nodeId, key, QString::fromUtf8(
                            QJsonDocument(updated).toJson(QJsonDocument::Compact)));
                    };
                    QWidget *field = nullptr;
                    const QString parameterType = QString::fromUtf8(def.type);
                    if (def.choice_count) {
                        auto *choices = new QComboBox(inspector_);
                        for (size_t c = 0; c < def.choice_count; ++c)
                            choices->addItem(QString::fromUtf8(def.choices[c]));
                        choices->setCurrentText(current.toString());
                        connect(choices, &QComboBox::currentTextChanged, this,
                                [submit](const QString &next) { submit(next); });
                        field = choices;
                    } else if (std::strcmp(def.type, "boolean") == 0) {
                        auto *check = new QCheckBox(inspector_);
                        check->setChecked(current.toBool());
                        connect(check, &QCheckBox::toggled, this,
                                [submit](bool checked) { submit(checked); });
                        field = check;
                    } else {
                        QString text = current.isString() ? parameterEditorText(current.toString(), parameterType)
                            : QString::fromUtf8(QJsonDocument(QJsonArray{current})
                                .toJson(QJsonDocument::Compact));
                        if (!current.isString()) text = text.mid(1, text.size() - 2);
                        auto *line = new QLineEdit(text, inspector_);
                        if (def.has_minimum) line->setToolTip(tr("Minimum: %1").arg(def.minimum));
                        connect(line, &QLineEdit::editingFinished, this, [this, line, parameterType, submit] {
                            const QJsonValue parsed = parameterEditorValue(parameterType, line->text());
                            if (parsed.isUndefined()) {
                                QMessageBox::warning(this, tr("Parameter rejected"), tr("Invalid parameter value"));
                                return;
                            }
                            submit(parsed);
                        });
                        field = line;
                    }
                    field->setObjectName(QStringLiteral("stereotypeParameter_%1_%2").arg(key, parameterKey));
                    auto *controls = new QWidget(inspector_);
                    auto *controlsLayout = new QHBoxLayout(controls);
                    controlsLayout->setContentsMargins(0, 0, 0, 0);
                    controlsLayout->addWidget(field, 1);
                    auto *binding = new QLineEdit(nestedBinding, controls);
                    binding->setObjectName(QStringLiteral("stereotypeParameter_%1_%2_binding").arg(key, parameterKey));
                    binding->setMaximumWidth(190);
                    binding->setPlaceholderText(tr("$name or $(Lua expression)"));
                    connect(binding, &QLineEdit::editingFinished, this,
                            [this, binding, parameterType, submit, nestedBinding, current] {
                        const QString text = binding->text();
                        if (text.isEmpty()) {
                            if (!nestedBinding.isEmpty()) submit(current);
                            return;
                        }
                        const QJsonValue next = parameterEditorValue(parameterType, text);
                        if (next.isUndefined()) {
                            QMessageBox::warning(this, tr("Parameter rejected"), tr("Enter $name or $(Lua expression)."));
                            return;
                        }
                        submit(next);
                    });
                    controlsLayout->addWidget(binding);
                    inspector_->setItemWidget(fieldRow, 1, controls);
                }
                editor = combo;
            } else if (type == QStringLiteral("boolean") || type == QStringLiteral("bool")) {
                auto *check = new QCheckBox(inspector_);
                check->setChecked((hasBinding ? resolvedValue : value) == QStringLiteral("true"));
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
            if (editor && (hasBinding || definition.choice_count > 0 || type == QStringLiteral("boolean") ||
                           type == QStringLiteral("bool") || type == QStringLiteral("stereotype"))) {
                auto *container = new QWidget(inspector_);
                auto *layout = new QVBoxLayout(container);
                layout->setContentsMargins(2, 2, 2, 2);
                layout->setSpacing(3);
                if (hasBinding && type != QStringLiteral("boolean") && type != QStringLiteral("bool") &&
                    type != QStringLiteral("stereotype") && definition.choice_count == 0) {
                    // Numeric and free-text editors already accept binding syntax directly.
                    layout->addWidget(editor);
                } else {
                    auto *controls = new QWidget(container);
                    auto *rowLayout = new QHBoxLayout(controls);
                    rowLayout->setContentsMargins(0, 0, 0, 0);
                    rowLayout->addWidget(editor, 1);
                    auto *binding = new QLineEdit(hasBinding ? value : QString(), controls);
                    binding->setObjectName(QStringLiteral("parameterBinding_%1").arg(key));
                    binding->setPlaceholderText(tr("$name or $(Lua expression)"));
                    binding->setMaximumWidth(190);
                    connect(binding, &QLineEdit::editingFinished, this,
                            [this, binding, nodeId, key, hasBinding, resolvedValue] {
                        if (!binding->text().isEmpty()) editNodeParameter(nodeId, key, binding->text());
                        else if (hasBinding && !resolvedValue.isEmpty()) editNodeParameter(nodeId, key, resolvedValue);
                    });
                    rowLayout->addWidget(binding);
                    layout->addWidget(controls);
                }
                auto *preview = new QLabel(container);
                preview->setObjectName(QStringLiteral("parameterPreview_%1").arg(key));
                preview->setWordWrap(true);
                preview->setStyleSheet(QStringLiteral("color: #596b7f; font-size: 10pt;"));
                if (!previewText && previewError[0]) {
                    preview->setText(tr("Resolved value: %1").arg(QString::fromUtf8(previewError)));
                    preview->setStyleSheet(QStringLiteral("color: #a22; font-size: 10pt;"));
                } else {
                    preview->setText(tr("Resolved: %1").arg(resolvedValue));
                }
                layout->addWidget(preview);
                editor = container;
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
    auto *operations = new QTreeWidgetItem(resources_, {tr("Operations")});
    operations->setExpanded(true);
    char error[ErrorCapacity] = {};
    char *operationText = nn_app_operations_json(application_.get(), error, sizeof(error));
    if (operationText) {
        const QJsonDocument operationDocument = QJsonDocument::fromJson(QByteArray(operationText));
        nn_app_free_text(operationText);
        if (operationDocument.isArray()) {
            const NNModel *model = nn_app_model(application_.get());
            for (const QJsonValue &value : operationDocument.array()) {
                const QJsonObject operation = value.toObject();
                QString state = tr("Ready");
                for (const QString &direction : {QStringLiteral("input"), QStringLiteral("output")}) {
                    const QJsonObject endpoint = operation.value(direction).toObject();
                    const QByteArray nodeId = endpoint.value(QStringLiteral("node")).toString().toUtf8();
                    const NNNode *node = nn_model_find_node(model, nodeId.constData());
                    bool found = false;
                    const bool output = direction == QStringLiteral("output");
                    const NNPackage *nodePackage = node
                        ? nn_catalog_find(catalog, node->package_id, node->package_version) : nullptr;
                    const bool inputNode = nodePackage && nodePackage->kind &&
                        std::strcmp(nodePackage->kind, "input") == 0;
                    const bool portOutput = output || (direction == QStringLiteral("input") && inputNode);
                    for (size_t p = 0; node && p < nn_app_port_count(application_.get(), node->id, portOutput); ++p) {
                        char port[256] = {};
                        if (nn_app_port_id(application_.get(), node->id, portOutput, p, port, sizeof(port)) &&
                            endpoint.value(QStringLiteral("handle")).toString() == QString::fromUtf8(port))
                            found = true;
                    }
                    if (!node || (node->scope_id && node->scope_id[0]) || !found) state = tr("Stale endpoint");
                }
                auto *item = new QTreeWidgetItem(operations, {QStringLiteral("%1  ·  %2")
                    .arg(operation.value(QStringLiteral("name")).toString(), state)});
                const QJsonObject input = operation.value(QStringLiteral("input")).toObject();
                const QJsonObject output = operation.value(QStringLiteral("output")).toObject();
                item->setToolTip(0, QStringLiteral("%1 (%2) → %3 (%4)")
                    .arg(input.value(QStringLiteral("node")).toString(), input.value(QStringLiteral("codec")).toString(),
                         output.value(QStringLiteral("node")).toString(), output.value(QStringLiteral("codec")).toString()));
            }
        }
    } else {
        new QTreeWidgetItem(operations, {tr("Unable to read operations: %1").arg(QString::fromUtf8(error))});
    }
    auto *metavariables = new QTreeWidgetItem(resources_, {tr("Metavariables")});
    metavariables->setExpanded(true);
    char variableError[ErrorCapacity] = {};
    char *variableText = nn_app_metavariables_json(application_.get(), variableError, sizeof(variableError));
    if (variableText) {
        const QJsonDocument variableDocument = QJsonDocument::fromJson(QByteArray(variableText));
        nn_app_free_text(variableText);
        for (const QJsonValue &entry : variableDocument.array()) {
            const QJsonObject variable = entry.toObject();
            const QString name = variable.value(QStringLiteral("name")).toString();
            const QString type = variable.value(QStringLiteral("type")).toString();
            QString value = variable.value(QStringLiteral("value")).isString()
                ? variable.value(QStringLiteral("value")).toString()
                : QString::fromUtf8(QJsonDocument(QJsonArray{variable.value(QStringLiteral("value"))})
                    .toJson(QJsonDocument::Compact)).mid(1).chopped(1);
            new QTreeWidgetItem(metavariables, {QStringLiteral("$%1  [%2] = %3").arg(name, type, value)});
        }
    } else {
        new QTreeWidgetItem(metavariables, {tr("Unable to read metavariables: %1").arg(QString::fromUtf8(variableError))});
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
