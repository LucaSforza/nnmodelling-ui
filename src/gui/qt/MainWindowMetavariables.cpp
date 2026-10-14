#include "MainWindow.hpp"

#include "MainWindowUtils.hpp"
#include "application/application.h"
#include "project/project.h"
#include "catalog/catalog.h"

#include <QComboBox>
#include <QDialog>
#include <QDialogButtonBox>
#include <QFormLayout>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMessageBox>
#include <QPushButton>
#include <QPlainTextEdit>
#include <QVBoxLayout>

#include <algorithm>
#include <cstring>

using namespace MainWindowUtils;

namespace {
QString compactJson(const QJsonValue &value)
{
    return QString::fromUtf8(QJsonDocument(QJsonArray{value}).toJson(QJsonDocument::Compact)).mid(1).chopped(1);
}

QJsonObject packageDefaults(const NNPackage *package)
{
    QJsonObject defaults;
    for (size_t i = 0; package && i < package->parameter_count; ++i) {
        const NNParameterDef &definition = package->parameters[i];
        if (!definition.key || !definition.type) continue;
        QJsonValue value;
        if (definition.has_default) {
            switch (definition.default_value.type) {
            case NN_PARAMETER_BOOLEAN: value = definition.default_value.as.boolean; break;
            case NN_PARAMETER_INTEGER: value = double(definition.default_value.as.integer); break;
            case NN_PARAMETER_NUMBER: value = definition.default_value.as.number; break;
            case NN_PARAMETER_STRING: value = QString::fromUtf8(definition.default_value.as.string); break;
            case NN_PARAMETER_JSON: {
                const QJsonDocument parsed = QJsonDocument::fromJson(definition.default_value.as.string);
                value = parsed.isArray() ? QJsonValue(parsed.array()) : QJsonValue(parsed.object());
                break;
            }
            default: break;
            }
        } else if (std::strcmp(definition.type, "integer") == 0)
            value = definition.has_minimum ? std::max(1.0, definition.minimum) : 1.0;
        else if (std::strcmp(definition.type, "number") == 0)
            value = definition.has_minimum ? definition.minimum : 0.0;
        else if (std::strcmp(definition.type, "boolean") == 0)
            value = false;
        else if (definition.choice_count)
            value = QString::fromUtf8(definition.choices[0]);
        else if (std::strcmp(definition.type, "string") == 0 || std::strcmp(definition.type, "dtype") == 0)
            value = QString();
        else if (std::strcmp(definition.type, "json") == 0)
            value = QJsonArray{};
        defaults.insert(QString::fromUtf8(definition.key), value);
    }
    return defaults;
}

class MetavariableForm final : public QDialog {
public:
    MetavariableForm(const QJsonObject &initial, const NNCatalog *catalog, QWidget *parent)
        : QDialog(parent)
    {
        setWindowTitle(tr("Metavariable"));
        auto *layout = new QVBoxLayout(this);
        auto *form = new QFormLayout;
        name_ = new QLineEdit(initial.value(QStringLiteral("name")).toString(), this);
        name_->setObjectName(QStringLiteral("metavariableName"));
        type_ = new QComboBox(this);
        type_->setObjectName(QStringLiteral("metavariableType"));
        for (const QString &type : {QStringLiteral("boolean"), QStringLiteral("integer"),
                                    QStringLiteral("number"), QStringLiteral("string"),
                                    QStringLiteral("dtype"), QStringLiteral("json"),
                                    QStringLiteral("stereotype")})
            type_->addItem(type, type);
        type_->setCurrentIndex(qMax(0, type_->findData(initial.value(QStringLiteral("type")).toString())));
        value_ = new QLineEdit(this);
        value_->setObjectName(QStringLiteral("metavariableValue"));
        value_->setPlaceholderText(tr("Literal value; JSON arrays use JSON syntax"));
        const QJsonValue initialValue = initial.value(QStringLiteral("value"));
        value_->setText(initialValue.isString() ? initialValue.toString() : compactJson(initialValue));
        stereotype_ = new QComboBox(this);
        stereotype_->setObjectName(QStringLiteral("metavariableStereotype"));
        stereotype_->addItem(tr("Choose package"), QJsonObject());
        for (size_t i = 0; catalog && i < nn_catalog_count(catalog); ++i) {
            const NNPackage *package = nn_catalog_at(catalog, i);
            if (!package || !package->id || !package->version || !package->kind ||
                std::strcmp(package->kind, "subflow") == 0) continue;
            QJsonObject ref{{QStringLiteral("id"), QString::fromUtf8(package->id)},
                            {QStringLiteral("version"), QString::fromUtf8(package->version)},
                            {QStringLiteral("parameters"), packageDefaults(package)}};
            stereotype_->addItem(QStringLiteral("%1  [%2@%3]")
                .arg(package->name ? QString::fromUtf8(package->name) : QString::fromUtf8(package->id),
                     QString::fromUtf8(package->id), QString::fromUtf8(package->version)), ref);
        }
        if (initialValue.isObject()) {
            const QJsonObject object = initialValue.toObject();
            const QString id = object.value(QStringLiteral("id")).toString();
            const QString version = object.value(QStringLiteral("version")).toString();
            for (int i = 1; i < stereotype_->count(); ++i) {
                QJsonObject candidate = stereotype_->itemData(i).toJsonObject();
                if (candidate.value(QStringLiteral("id")).toString() == id &&
                    candidate.value(QStringLiteral("version")).toString() == version) {
                    candidate.insert(QStringLiteral("parameters"), object.value(QStringLiteral("parameters")).toObject());
                    stereotype_->setItemData(i, candidate);
                    stereotype_->setCurrentIndex(i);
                    break;
                }
            }
        }
        stereotypeValue_ = new QPlainTextEdit(this);
        stereotypeValue_->setObjectName(QStringLiteral("metavariableStereotypeValue"));
        stereotypeValue_->setMaximumHeight(95);
        const QJsonObject selectedReference = initialValue.isObject() ? initialValue.toObject()
            : stereotype_->currentData().toJsonObject();
        stereotypeValue_->setPlainText(QString::fromUtf8(
            QJsonDocument(selectedReference).toJson(QJsonDocument::Compact)));
        stereotype_->setProperty("referenceIndex", stereotype_->currentIndex());
        connect(stereotype_, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int index) {
            const int previous = stereotype_->property("referenceIndex").toInt();
            if (previous > 0) {
                QJsonParseError error;
                const QJsonDocument edited = QJsonDocument::fromJson(stereotypeValue_->toPlainText().toUtf8(), &error);
                if (error.error == QJsonParseError::NoError && edited.isObject()) {
                    QJsonObject reference = stereotype_->itemData(previous).toJsonObject();
                    reference.insert(QStringLiteral("parameters"),
                                     edited.object().value(QStringLiteral("parameters")).toObject());
                    stereotype_->setItemData(previous, reference);
                }
            }
            stereotypeValue_->setPlainText(QString::fromUtf8(
                QJsonDocument(stereotype_->itemData(index).toJsonObject()).toJson(QJsonDocument::Compact)));
            stereotype_->setProperty("referenceIndex", index);
        });
        form->addRow(tr("Name"), name_);
        form->addRow(tr("Type"), type_);
        form->addRow(tr("Value"), value_);
        form->addRow(tr("Catalog reference"), stereotype_);
        form->addRow(tr("Reference value"), stereotypeValue_);
        layout->addLayout(form);
        auto *buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
        layout->addWidget(buttons);
        connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
        connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
        auto updateFields = [this, form] {
            const QString type = type_->currentData().toString();
            value_->setVisible(type != QStringLiteral("stereotype"));
            if (QWidget *label = form->labelForField(value_))
                label->setVisible(value_->isVisible());
            stereotype_->setVisible(type == QStringLiteral("stereotype"));
            if (QWidget *label = form->labelForField(stereotype_))
                label->setVisible(stereotype_->isVisible());
            stereotypeValue_->setVisible(type == QStringLiteral("stereotype"));
            if (QWidget *label = form->labelForField(stereotypeValue_))
                label->setVisible(stereotypeValue_->isVisible());
            if (type == QStringLiteral("boolean")) {
                value_->setPlaceholderText(tr("true or false"));
            } else if (type == QStringLiteral("json")) {
                value_->setPlaceholderText(tr("JSON array, for example [1, 2]"));
            } else {
                value_->setPlaceholderText(tr("Literal value"));
            }
        };
        connect(type_, qOverload<int>(&QComboBox::currentIndexChanged), this, updateFields);
        updateFields();
    }

    QJsonObject value() const
    {
        const QString type = type_->currentData().toString();
        QJsonValue parsed;
        if (type == QStringLiteral("stereotype")) {
            QJsonParseError error;
            const QJsonDocument document = QJsonDocument::fromJson(stereotypeValue_->toPlainText().toUtf8(), &error);
            if (error.error == QJsonParseError::NoError && document.isObject()) parsed = document.object();
        } else if (type == QStringLiteral("string") || type == QStringLiteral("dtype")) {
            parsed = value_->text();
        } else if (type == QStringLiteral("json")) {
            QJsonParseError error;
            const QJsonDocument document = QJsonDocument::fromJson(value_->text().toUtf8(), &error);
            if (error.error == QJsonParseError::NoError && document.isArray()) parsed = document.array();
        } else {
            const QByteArray bytes = (type == QStringLiteral("json") ? value_->text()
                : QStringLiteral("[") + value_->text() + QStringLiteral("]")).toUtf8();
            const QJsonDocument document = QJsonDocument::fromJson(bytes);
            if (document.isArray() && document.array().size() == 1) parsed = document.array().at(0);
        }
        return {{QStringLiteral("name"), name_->text().trimmed()},
                {QStringLiteral("type"), type}, {QStringLiteral("value"), parsed}};
    }

private:
    QLineEdit *name_ = nullptr;
    QComboBox *type_ = nullptr;
    QLineEdit *value_ = nullptr;
    QComboBox *stereotype_ = nullptr;
    QPlainTextEdit *stereotypeValue_ = nullptr;
};
}

void MainWindow::manageMetavariables()
{
    char error[ErrorCapacity] = {};
    char *text = nn_app_metavariables_json(application_.get(), error, sizeof(error));
    if (!text) {
        QMessageBox::warning(this, tr("Metavariables unavailable"), QString::fromUtf8(error));
        return;
    }
    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(QByteArray(text), &parseError);
    nn_app_free_text(text);
    if (!document.isArray()) {
        QMessageBox::warning(this, tr("Metavariables unavailable"), tr("The project returned invalid metavariable data."));
        return;
    }
    QJsonArray staged = document.array();
    QDialog dialog(this);
    dialog.setWindowTitle(tr("Manage metavariables"));
    dialog.resize(580, 430);
    auto *layout = new QVBoxLayout(&dialog);
    auto *list = new QListWidget(&dialog);
    list->setObjectName(QStringLiteral("metavariablesList"));
    auto populate = [&] {
        list->clear();
        for (const QJsonValue &entry : staged) {
            const QJsonObject item = entry.toObject();
            const QString value = item.value(QStringLiteral("type")).toString() == QStringLiteral("stereotype")
                ? QStringLiteral("%1@%2").arg(item.value(QStringLiteral("value")).toObject().value(QStringLiteral("id")).toString(),
                                               item.value(QStringLiteral("value")).toObject().value(QStringLiteral("version")).toString())
                : compactJson(item.value(QStringLiteral("value")));
            auto *row = new QListWidgetItem(QStringLiteral("$%1  ·  %2  ·  %3")
                .arg(item.value(QStringLiteral("name")).toString(), item.value(QStringLiteral("type")).toString(), value), list);
            row->setData(Qt::UserRole, item);
        }
    };
    populate();
    layout->addWidget(list, 1);
    auto *actions = new QHBoxLayout;
    auto *newButton = new QPushButton(tr("New"), &dialog);
    auto *editButton = new QPushButton(tr("Edit"), &dialog);
    auto *removeButton = new QPushButton(tr("Remove"), &dialog);
    newButton->setObjectName(QStringLiteral("newMetavariable"));
    editButton->setObjectName(QStringLiteral("editMetavariable"));
    removeButton->setObjectName(QStringLiteral("removeMetavariable"));
    actions->addWidget(newButton);
    actions->addWidget(editButton);
    actions->addWidget(removeButton);
    actions->addStretch();
    layout->addLayout(actions);
    auto edit = [&](int row) {
        const QJsonObject initial = row >= 0 ? staged.at(row).toObject() : QJsonObject{};
        const NNProject *project = nn_app_project(application_.get());
        const NNCatalog *catalog = project ? nn_project_catalog(project) : nullptr;
        MetavariableForm form(initial, catalog, &dialog);
        if (form.exec() != QDialog::Accepted) return;
        const QJsonObject next = form.value();
        if (row >= 0) staged.replace(row, next);
        else staged.append(next);
        populate();
        list->setCurrentRow(row >= 0 ? row : staged.size() - 1);
    };
    connect(newButton, &QPushButton::clicked, &dialog, [&] { edit(-1); });
    connect(editButton, &QPushButton::clicked, &dialog, [&] {
        if (list->currentRow() >= 0) edit(list->currentRow());
    });
    connect(removeButton, &QPushButton::clicked, &dialog, [&] {
        const int row = list->currentRow();
        if (row < 0) return;
        staged.removeAt(row);
        populate();
    });
    auto *buttons = new QDialogButtonBox(QDialogButtonBox::Save | QDialogButtonBox::Cancel, &dialog);
    buttons->setObjectName(QStringLiteral("metavariablesButtons"));
    layout->addWidget(buttons);
    connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    connect(buttons, &QDialogButtonBox::accepted, &dialog, [&] {
        const QByteArray json = QJsonDocument(staged).toJson(QJsonDocument::Compact);
        char setError[ErrorCapacity] = {};
        if (!nn_app_set_metavariables_json(application_.get(), json.constData(), setError, sizeof(setError))) {
            QMessageBox::warning(&dialog, tr("Metavariables rejected"), QString::fromUtf8(setError));
            return;
        }
        dialog.accept();
    });
    if (dialog.exec() == QDialog::Accepted) refreshAll();
}
