#include "MainWindow.hpp"

#include "MainWindowUtils.hpp"
#include "application/application.h"
#include "catalog/catalog.h"
#include "inference/inference.h"
#include "model/model.h"
#include "project/project.h"

#include <QComboBox>
#include <QDialog>
#include <QDialogButtonBox>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMessageBox>
#include <QPushButton>
#include <QStringList>
#include <QVBoxLayout>

#include <cstring>

extern "C" char *nn_app_operations_json(const NNApplication *app, char *error, size_t capacity);
extern "C" bool nn_app_set_operations_json(NNApplication *app, const char *json,
                                              char *error, size_t capacity);

using namespace MainWindowUtils;

namespace {
QString tensorText(const NNInferenceResult *result, const char *handle) {
    if (!result) return QString();
    for (size_t i = 0; i < result->output_count; ++i) {
        const NNInferenceTensor &tensor = result->outputs[i];
        if (std::strcmp(tensor.handle_id ? tensor.handle_id : "", handle ? handle : "") != 0) continue;
        QStringList dimensions;
        for (size_t j = 0; j < tensor.dimension_count; ++j)
            dimensions.push_back(QString::fromUtf8(tensor.dimensions[j] ? tensor.dimensions[j] : "?"));
        return QStringLiteral("%1[%2]").arg(QString::fromUtf8(tensor.dtype ? tensor.dtype : "?"),
                                               dimensions.join(QStringLiteral(", ")));
    }
    return QString();
}

const NNInferenceResult *resultFor(const NNInferenceReport *report, const char *nodeId) {
    for (size_t i = 0; report && i < nn_inference_count(report); ++i) {
        const NNInferenceResult *result = nn_inference_at(report, i);
        if (result && std::strcmp(result->node_id ? result->node_id : "", nodeId ? nodeId : "") == 0)
            return result;
    }
    return nullptr;
}

QString endpointSignature(const NNModel *model, const NNInferenceReport *report,
                          const QString &nodeId, const QString &handle, bool input) {
    const QByteArray id = nodeId.toUtf8();
    const NNNode *node = nn_model_find_node(model, id.constData());
    if (!node) return QStringLiteral("stale endpoint");
    if (input) {
        const QString direct = tensorText(resultFor(report, node->id), handle.toUtf8().constData());
        if (!direct.isEmpty()) return direct;
        for (size_t i = 0; i < nn_model_edge_count(model); ++i) {
            const NNEdge *edge = nn_model_edge_at(model, i);
            if (std::strcmp(edge->target_id, node->id) == 0 &&
                std::strcmp(edge->target_handle_id, handle.toUtf8().constData()) == 0) {
                const NNInferenceResult *source = resultFor(report, edge->source_id);
                const QString text = tensorText(source, edge->source_handle_id);
                return text.isEmpty() ? QStringLiteral("unresolved") : text;
            }
        }
        return QStringLiteral("unresolved");
    }
    const QString text = tensorText(resultFor(report, node->id), handle.toUtf8().constData());
    return text.isEmpty() ? QStringLiteral("unresolved") : text;
}

QString operationStatus(const NNApplication *app, const QJsonObject &operation) {
    const NNModel *model = nn_app_model(app);
    const NNProject *project = nn_app_project(app);
    const NNCatalog *catalog = project ? nn_project_catalog(project) : nullptr;
    const auto endpointValid = [model, app, catalog](const QJsonObject &endpoint, bool operationInput) {
        const QString nodeId = endpoint.value(QStringLiteral("node")).toString();
        const QString handle = endpoint.value(QStringLiteral("handle")).toString();
        const QByteArray id = nodeId.toUtf8();
        const NNNode *node = nn_model_find_node(model, id.constData());
        if (!node || (node->scope_id && node->scope_id[0]) || handle.isEmpty()) return false;
        const NNPackage *package = nn_catalog_find(catalog, node->package_id, node->package_version);
        const bool inputNode = package && package->kind && std::strcmp(package->kind, "input") == 0;
        const bool portOutput = !operationInput || inputNode;
        for (size_t i = 0; i < nn_app_port_count(app, node->id, portOutput); ++i) {
            char port[256] = {};
            if (nn_app_port_id(app, node->id, portOutput, i, port, sizeof(port)) && handle == QString::fromUtf8(port))
                return true;
        }
        return false;
    };
    const QJsonObject inputEndpoint = operation.value(QStringLiteral("input")).toObject();
    const QJsonObject outputEndpoint = operation.value(QStringLiteral("output")).toObject();
    if (!endpointValid(inputEndpoint, true) || !endpointValid(outputEndpoint, false))
        return QObject::tr("Stale endpoint");
    char error[ErrorCapacity] = {};
    const NNInferenceReport *analysis = nn_app_analysis(const_cast<NNApplication *>(app), error, sizeof(error));
    const QJsonObject input = operation.value(QStringLiteral("input")).toObject();
    const QJsonObject output = operation.value(QStringLiteral("output")).toObject();
    const QString in = endpointSignature(model, analysis, input.value(QStringLiteral("node")).toString(),
                                         input.value(QStringLiteral("handle")).toString(), true);
    const QString out = endpointSignature(model, analysis, output.value(QStringLiteral("node")).toString(),
                                          output.value(QStringLiteral("handle")).toString(), false);
    return in == QStringLiteral("unresolved") || out == QStringLiteral("unresolved")
        ? QObject::tr("Unresolved") : QObject::tr("Ready");
}

class EndpointFields final : public QWidget {
public:
    EndpointFields(NNApplication *app, bool input, const QJsonObject &initial, QWidget *parent = nullptr)
        : QWidget(parent), app_(app), input_(input) {
        auto *layout = new QHBoxLayout(this);
        layout->setContentsMargins(0, 0, 0, 0);
        node_ = new QComboBox(this);
        handle_ = new QComboBox(this);
        node_->setObjectName(input ? QStringLiteral("operationInputNode") : QStringLiteral("operationOutputNode"));
        handle_->setObjectName(input ? QStringLiteral("operationInputHandle") : QStringLiteral("operationOutputHandle"));
        layout->addWidget(node_, 1);
        layout->addWidget(handle_, 1);
        const NNModel *model = nn_app_model(app_);
        for (size_t i = 0; model && i < nn_model_node_count(model); ++i) {
            const NNNode *node = nn_model_node_at(model, i);
            if (!node || (node->scope_id && node->scope_id[0])) continue;
            const NNProject *project = nn_app_project(app_);
            const NNPackage *package = project ? nn_catalog_find(nn_project_catalog(project),
                node->package_id, node->package_version) : nullptr;
            const QString kind = package && package->kind ? QString::fromUtf8(package->kind) : QString();
            if (input_ && (kind == QStringLiteral("output") || kind == QStringLiteral("loss-output"))) continue;
            if (!input_ && (kind == QStringLiteral("input") || kind == QStringLiteral("output") ||
                            kind == QStringLiteral("loss-output"))) continue;
            node_->addItem(QStringLiteral("%1  [%2]").arg(QString::fromUtf8(node->label),
                                                           QString::fromUtf8(node->id)), QString::fromUtf8(node->id));
        }
        const QString oldNode = initial.value(QStringLiteral("node")).toString();
        const int oldNodeIndex = node_->findData(oldNode);
        if (oldNodeIndex < 0 && !oldNode.isEmpty())
            node_->addItem(QObject::tr("Invalid node: %1").arg(oldNode), oldNode);
        node_->setCurrentIndex(node_->findData(oldNode));
        QObject::connect(node_, qOverload<int>(&QComboBox::currentIndexChanged), this,
                         [this](int) { refreshHandles(QString()); });
        refreshHandles(initial.value(QStringLiteral("handle")).toString());
    }

    QJsonObject value() const {
        return {{QStringLiteral("node"), node_->currentData().toString()},
                {QStringLiteral("handle"), handle_->currentData().toString()}};
    }

    QString signature(const NNInferenceReport *report) const {
        return endpointSignature(nn_app_model(app_), report, node_->currentData().toString(),
                                 handle_->currentData().toString(), input_);
    }

private:
    void refreshHandles(const QString &preserve) {
        const QString currentNode = node_->currentData().toString();
        handle_->clear();
        const QByteArray nodeId = currentNode.toUtf8();
        const NNNode *node = nn_model_find_node(nn_app_model(app_), nodeId.constData());
        const NNProject *project = nn_app_project(app_);
        const NNPackage *package = node && project
            ? nn_catalog_find(nn_project_catalog(project), node->package_id, node->package_version) : nullptr;
        const bool output = !input_ || (package && package->kind &&
                                         std::strcmp(package->kind, "input") == 0);
        for (size_t i = 0; !currentNode.isEmpty() && i < nn_app_port_count(app_, nodeId.constData(), output); ++i) {
            char id[256] = {};
            if (nn_app_port_id(app_, nodeId.constData(), output, i, id, sizeof(id)))
                handle_->addItem(QString::fromUtf8(id), QString::fromUtf8(id));
        }
        if (!preserve.isEmpty() && handle_->findData(preserve) < 0)
            handle_->addItem(QObject::tr("Invalid handle: %1").arg(preserve), preserve);
        const int index = handle_->findData(preserve);
        if (index >= 0) handle_->setCurrentIndex(index);
    }

    NNApplication *app_;
    bool input_;
    QComboBox *node_ = nullptr;
    QComboBox *handle_ = nullptr;
};

class OperationForm final : public QDialog {
public:
    OperationForm(NNApplication *app, const QJsonObject &initial, QWidget *parent)
        : QDialog(parent), app_(app) {
        setWindowTitle(QObject::tr("Operation"));
        auto *layout = new QVBoxLayout(this);
        auto *form = new QFormLayout;
        name_ = new QLineEdit(initial.value(QStringLiteral("name")).toString(), this);
        name_->setObjectName(QStringLiteral("operationName"));
        inCodec_ = new QComboBox(this);
        outCodec_ = new QComboBox(this);
        inCodec_->setObjectName(QStringLiteral("operationInputCodec"));
        outCodec_->setObjectName(QStringLiteral("operationOutputCodec"));
        inCodec_->addItems({QStringLiteral("dataset"), QStringLiteral("tensor")});
        outCodec_->addItems({QStringLiteral("dataset"), QStringLiteral("tensor")});
        inCodec_->setCurrentText(initial.value(QStringLiteral("input")).toObject().value(QStringLiteral("codec")).toString(QStringLiteral("dataset")));
        outCodec_->setCurrentText(initial.value(QStringLiteral("output")).toObject().value(QStringLiteral("codec")).toString(QStringLiteral("dataset")));
        input_ = new EndpointFields(app_, true, initial.value(QStringLiteral("input")).toObject(), this);
        output_ = new EndpointFields(app_, false, initial.value(QStringLiteral("output")).toObject(), this);
        form->addRow(QObject::tr("Method name"), name_);
        form->addRow(QObject::tr("Input endpoint"), input_);
        form->addRow(QObject::tr("Input codec"), inCodec_);
        form->addRow(QObject::tr("Output endpoint"), output_);
        form->addRow(QObject::tr("Output codec"), outCodec_);
        signature_ = new QLabel(this);
        signature_->setObjectName(QStringLiteral("operationSignature"));
        signature_->setWordWrap(true);
        form->addRow(QObject::tr("Current signature"), signature_);
        layout->addLayout(form);
        auto *buttons = new QDialogButtonBox(QDialogButtonBox::Save | QDialogButtonBox::Cancel, this);
        layout->addWidget(buttons);
        QObject::connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
        QObject::connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
        char error[ErrorCapacity] = {};
        analysis_ = nn_app_analysis(app_, error, sizeof(error));
        const auto updateSignature = [this] {
            signature_->setText(QObject::tr("%1 → %2")
                .arg(input_->signature(analysis_), output_->signature(analysis_)));
        };
        updateSignature();
        QObject::connect(input_->findChild<QComboBox *>(QStringLiteral("operationInputNode")),
                         qOverload<int>(&QComboBox::currentIndexChanged), this, updateSignature);
        QObject::connect(input_->findChild<QComboBox *>(QStringLiteral("operationInputHandle")),
                         qOverload<int>(&QComboBox::currentIndexChanged), this, updateSignature);
        QObject::connect(output_->findChild<QComboBox *>(QStringLiteral("operationOutputNode")),
                         qOverload<int>(&QComboBox::currentIndexChanged), this, updateSignature);
        QObject::connect(output_->findChild<QComboBox *>(QStringLiteral("operationOutputHandle")),
                         qOverload<int>(&QComboBox::currentIndexChanged), this, updateSignature);
    }

    QJsonObject value() const {
        QJsonObject input = input_->value();
        QJsonObject output = output_->value();
        input.insert(QStringLiteral("codec"), inCodec_->currentText());
        output.insert(QStringLiteral("codec"), outCodec_->currentText());
        return {{QStringLiteral("name"), name_->text().trimmed()},
                {QStringLiteral("input"), input}, {QStringLiteral("output"), output}};
    }

private:
    NNApplication *app_;
    const NNInferenceReport *analysis_ = nullptr;
    QLineEdit *name_ = nullptr;
    QComboBox *inCodec_ = nullptr;
    QComboBox *outCodec_ = nullptr;
    EndpointFields *input_ = nullptr;
    EndpointFields *output_ = nullptr;
    QLabel *signature_ = nullptr;
};
}

void MainWindow::manageOperations() {
    if (!nn_app_project(application_.get())) return;
    char error[ErrorCapacity] = {};
    char *json = nn_app_operations_json(application_.get(), error, sizeof(error));
    if (!json) {
        QMessageBox::critical(this, tr("Operations unavailable"), QString::fromUtf8(error));
        return;
    }
    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(QByteArray(json), &parseError);
    nn_app_free_text(json);
    if (parseError.error != QJsonParseError::NoError || !document.isArray()) {
        QMessageBox::critical(this, tr("Operations unavailable"), tr("The project operations could not be read."));
        return;
    }
    QJsonArray operations = document.array();
    QDialog dialog(this);
    dialog.setWindowTitle(tr("Manage operations"));
    dialog.resize(680, 460);
    auto *layout = new QVBoxLayout(&dialog);
    auto *list = new QListWidget(&dialog);
    list->setObjectName(QStringLiteral("operationsList"));
    layout->addWidget(list, 1);
    auto refreshList = [&] {
        list->clear();
        for (const QJsonValue &value : operations) {
            const QJsonObject op = value.toObject();
            const QString input = op.value(QStringLiteral("input")).toObject().value(QStringLiteral("node")).toString();
            const QString output = op.value(QStringLiteral("output")).toObject().value(QStringLiteral("node")).toString();
            list->addItem(QStringLiteral("%1    %2    %3 → %4")
                .arg(op.value(QStringLiteral("name")).toString(), operationStatus(application_.get(), op), input, output));
        }
    };
    refreshList();
    auto *controls = new QHBoxLayout;
    auto *newButton = new QPushButton(tr("New…"), &dialog);
    newButton->setObjectName(QStringLiteral("newOperation"));
    auto *editButton = new QPushButton(tr("Edit…"), &dialog);
    editButton->setObjectName(QStringLiteral("editOperation"));
    auto *removeButton = new QPushButton(tr("Remove"), &dialog);
    removeButton->setObjectName(QStringLiteral("removeOperation"));
    controls->addWidget(newButton);
    controls->addWidget(editButton);
    controls->addWidget(removeButton);
    controls->addStretch();
    layout->addLayout(controls);
    auto editSelected = [&] {
        const int index = list->currentRow();
        if (index < 0 || index >= operations.size()) return;
        OperationForm form(application_.get(), operations.at(index).toObject(), &dialog);
        if (form.exec() == QDialog::Accepted) {
            operations.replace(index, form.value());
            refreshList();
            list->setCurrentRow(index);
        }
    };
    connect(newButton, &QPushButton::clicked, &dialog, [&] {
        OperationForm form(application_.get(), {}, &dialog);
        if (form.exec() == QDialog::Accepted) {
            operations.append(form.value());
            refreshList();
            list->setCurrentRow(list->count() - 1);
        }
    });
    connect(editButton, &QPushButton::clicked, &dialog, editSelected);
    connect(list, &QListWidget::itemDoubleClicked, &dialog, [editSelected] { editSelected(); });
    connect(removeButton, &QPushButton::clicked, &dialog, [&] {
        const int index = list->currentRow();
        if (index < 0 || index >= operations.size()) return;
        operations.removeAt(index);
        refreshList();
    });
    auto *buttons = new QDialogButtonBox(QDialogButtonBox::Save | QDialogButtonBox::Cancel, &dialog);
    layout->addWidget(buttons);
    connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    connect(buttons, &QDialogButtonBox::accepted, &dialog, [&] {
        const QByteArray updated = QJsonDocument(operations).toJson(QJsonDocument::Compact);
        char saveError[ErrorCapacity] = {};
        if (!nn_app_set_operations_json(application_.get(), updated.constData(), saveError, sizeof(saveError))) {
            QMessageBox::warning(&dialog, tr("Operations rejected"), QString::fromUtf8(saveError));
            return;
        }
        dialog.accept();
    });
    if (dialog.exec() == QDialog::Accepted) refreshAll();
}
