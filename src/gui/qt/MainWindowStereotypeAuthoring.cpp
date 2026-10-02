#include "MainWindow.hpp"
#include "MainWindowUtils.hpp"
#include "application/application.h"
#include "inference/inference.h"
#include <QCheckBox>
#include <QColor>
#include <QComboBox>
#include <QDialog>
#include <QDialogButtonBox>
#include <QFormLayout>
#include <QHeaderView>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QSet>
#include <QStatusBar>
#include <QTextBlock>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QTextCursor>
#include <QTextEdit>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonValue>
#include <cmath>

using namespace MainWindowUtils;

namespace {
QJsonValue parseDefault(QString text, const QString &type, bool *ok) {
    *ok = true;
    if (type == QStringLiteral("boolean")) { if (text == QStringLiteral("true")) return true; if (text == QStringLiteral("false")) return false; *ok = false; return {}; }
    if (type == QStringLiteral("integer")) { bool parsed = false; const qlonglong v = text.toLongLong(&parsed); *ok = parsed; return *ok ? QJsonValue(qint64(v)) : QJsonValue(); }
    if (type == QStringLiteral("number")) { bool parsed = false; const double v = text.toDouble(&parsed); *ok = parsed && std::isfinite(v); return *ok ? QJsonValue(v) : QJsonValue(); }
    if (type == QStringLiteral("json")) { const auto doc = QJsonDocument::fromJson(text.toUtf8()); *ok = doc.isArray(); return *ok ? QJsonValue(doc.array()) : QJsonValue(); }
    return text;
}
}

void MainWindow::createStereotype() {
    if (!nn_app_project(application_.get())) {
        statusBar()->showMessage(tr("Open a project before creating a stereotype"), 5000);
        return;
    }
    QDialog dialog(this);
    dialog.setWindowTitle(tr("Create stereotype — saves project"));
    dialog.setMinimumWidth(760);
    auto *layout = new QVBoxLayout(&dialog);
    auto *form = new QFormLayout;
    auto *id = new QLineEdit(&dialog); id->setObjectName("stereotypeId");
    auto *version = new QLineEdit(QStringLiteral("1.0.0"), &dialog); version->setObjectName("stereotypeVersion");
    auto *name = new QLineEdit(&dialog); name->setObjectName("stereotypeName");
    auto *description = new QLineEdit(&dialog); description->setObjectName("stereotypeDescription");
    auto *kind = new QComboBox(&dialog); kind->setObjectName("stereotypeKind");
    kind->addItems({"input", "layer", "join", "loss", "output", "loss-output", "subflow"});
    kind->setCurrentText(QStringLiteral("layer"));
    auto *color = new QLineEdit(QStringLiteral("#6b8fc4"), &dialog); color->setObjectName("stereotypeColor");
    form->addRow(tr("ID"), id); form->addRow(tr("Version"), version);
    form->addRow(tr("Name"), name); form->addRow(tr("Description"), description);
    form->addRow(tr("Kind"), kind); form->addRow(tr("Color"), color);
    layout->addLayout(form);

    auto *overrideOutputs = new QCheckBox(tr("Use explicit output handles instead of defaults"), &dialog);
    overrideOutputs->setObjectName(QStringLiteral("stereotypeOutputOverride"));
    layout->addWidget(overrideOutputs);
    auto *outputDefaults = new QLabel(&dialog);
    outputDefaults->setObjectName(QStringLiteral("stereotypeOutputDefaults"));
    auto updateOutputDefaults = [outputDefaults](const QString &selectedKind) {
        if (selectedKind == QStringLiteral("output") || selectedKind == QStringLiteral("loss-output"))
            outputDefaults->setText(QObject::tr("Default outputs: none (terminal)."));
        else if (selectedKind == QStringLiteral("loss"))
            outputDefaults->setText(QObject::tr("Default outputs: loss (handle ID: loss)."));
        else
            outputDefaults->setText(QObject::tr("Default outputs: output (handle ID: out)."));
    };
    updateOutputDefaults(kind->currentText());
    connect(kind, &QComboBox::currentTextChanged, &dialog, updateOutputDefaults);
    layout->addWidget(outputDefaults);
    layout->addWidget(new QLabel(tr("Explicit choices replace defaults; at most one output and one loss."), &dialog));
    auto *outputs = new QTableWidget(0, 2, &dialog);
    outputs->setObjectName(QStringLiteral("stereotypeOutputs"));
    outputs->setHorizontalHeaderLabels({tr("Output ID"), tr("Type (output or loss)")});
    outputs->horizontalHeader()->setStretchLastSection(true);
    outputs->setEnabled(false);
    layout->addWidget(outputs);
    connect(overrideOutputs, &QCheckBox::toggled, outputs, &QTableWidget::setEnabled);
    auto *outputActions = new QHBoxLayout;
    auto *addOutput = new QPushButton(tr("Add output row"), &dialog);
    auto *removeOutput = new QPushButton(tr("Remove output row"), &dialog);
    outputActions->addWidget(addOutput); outputActions->addWidget(removeOutput);
    outputActions->addStretch(); layout->addLayout(outputActions);
    connect(addOutput, &QPushButton::clicked, &dialog, [outputs] {
        addTableRow(outputs, {QString(), QStringLiteral("output")});
    });
    connect(removeOutput, &QPushButton::clicked, &dialog, [outputs] {
        if (outputs->currentRow() >= 0) outputs->removeRow(outputs->currentRow());
    });

    layout->addWidget(new QLabel(tr("Parameters — types: boolean, integer, number, string, dtype or JSON array; position: top, bottom or blank."), &dialog));
    auto *parameters = new QTableWidget(0, 7, &dialog);
    parameters->setObjectName("stereotypeParameters");
    parameters->setHorizontalHeaderLabels({tr("Key"), tr("Type"), tr("Default"), tr("Minimum"), tr("Choices (comma separated)"), tr("Position"), tr(" ")});
    parameters->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
    parameters->horizontalHeader()->setStretchLastSection(true);
    layout->addWidget(parameters, 1);
    auto *parameterActions = new QHBoxLayout;
    auto *addParameter = new QPushButton(tr("Add parameter row"), &dialog);
    auto *removeParameter = new QPushButton(tr("Remove selected row"), &dialog);
    parameterActions->addWidget(addParameter); parameterActions->addWidget(removeParameter);
    parameterActions->addStretch(); layout->addLayout(parameterActions);
    connect(addParameter, &QPushButton::clicked, &dialog, [parameters] {
        addTableRow(parameters, {QString(), QStringLiteral("number"), QStringLiteral("0"), QString(), QString(), QString(), QString()});
    });
    connect(removeParameter, &QPushButton::clicked, &dialog, [parameters] {
        if (parameters->currentRow() >= 0) parameters->removeRow(parameters->currentRow());
    });

    layout->addWidget(new QLabel(tr("Dependencies — package ID and version constraint."), &dialog));
    auto *dependencies = new QTableWidget(0, 2, &dialog);
    dependencies->setObjectName("stereotypeDependencies");
    dependencies->setHorizontalHeaderLabels({tr("Package ID"), tr("Version constraint")});
    dependencies->horizontalHeader()->setStretchLastSection(true);
    layout->addWidget(dependencies);
    auto *dependencyActions = new QHBoxLayout;
    auto *addDependency = new QPushButton(tr("Add dependency"), &dialog);
    auto *removeDependency = new QPushButton(tr("Remove dependency"), &dialog);
    dependencyActions->addWidget(addDependency); dependencyActions->addWidget(removeDependency);
    dependencyActions->addStretch(); layout->addLayout(dependencyActions);
    connect(addDependency, &QPushButton::clicked, &dialog, [dependencies] {
        addTableRow(dependencies, {QString(), QStringLiteral("0.1.0")});
    });
    connect(removeDependency, &QPushButton::clicked, &dialog, [dependencies] {
        if (dependencies->currentRow() >= 0) dependencies->removeRow(dependencies->currentRow());
    });

    layout->addWidget(new QLabel(tr("Optional Lua inference (pass-through shape rule by default)."), &dialog));
    auto *lua = new QPlainTextEdit(&dialog);
    lua->setObjectName("stereotypeLua");
    const QString passThroughRule = QStringLiteral(
        "return function(context, parameters, services)\n"
        "  if not context.inputs[1] then\n"
        "    return { status = 'unresolved', message = 'Input missing' }\n"
        "  end\n"
        "  return { status = 'success', output = context.inputs[1] }\n"
        "end");
    lua->setPlaceholderText(passThroughRule);
    lua->setPlainText(passThroughRule);
    lua->setMinimumHeight(90);
    layout->addWidget(lua);
    auto *luaError = new QLabel(&dialog);
    luaError->setObjectName(QStringLiteral("stereotypeLuaError"));
    luaError->setWordWrap(true);
    luaError->setStyleSheet(QStringLiteral("color: #a12b2b;"));
    layout->addWidget(luaError);
    auto *formError = new QLabel(&dialog);
    formError->setObjectName(QStringLiteral("stereotypeError"));
    formError->setWordWrap(true);
    formError->setStyleSheet(QStringLiteral("color: #a12b2b;"));
    layout->addWidget(formError);
    auto *buttons = new QDialogButtonBox(QDialogButtonBox::Save | QDialogButtonBox::Cancel, &dialog);
    buttons->button(QDialogButtonBox::Save)->setText(tr("Create and save project"));
    connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    layout->addWidget(buttons);
    while (dialog.exec() == QDialog::Accepted) {
    luaError->clear();
    formError->clear();

    QJsonObject def;
    def.insert("name", name->text().trimmed()); def.insert("description", description->text().trimmed());
    def.insert("kind", kind->currentText());
    if (overrideOutputs->isChecked()) {
        QJsonArray definitions;
        QSet<QString> ids, types;
        for (int row = 0; row < outputs->rowCount(); ++row) {
            const QString outputId = cellText(outputs, row, 0);
            const QString type = cellText(outputs, row, 1);
            if (outputId.isEmpty() || (type != QStringLiteral("output") && type != QStringLiteral("loss")) ||
                ids.contains(outputId) || types.contains(type)) {
                formError->setText(tr("Output IDs must be nonempty and unique; choose at most one row of each type."));
                break;
            }
            ids.insert(outputId); types.insert(type);
            definitions.append(QJsonObject{{QStringLiteral("id"), outputId},
                                           {QStringLiteral("type"), type}});
        }
        if (!formError->text().isEmpty()) continue;
        if ((kind->currentText() == QStringLiteral("output") ||
             kind->currentText() == QStringLiteral("loss-output")) && !definitions.isEmpty()) {
            formError->setText(tr("Terminal kinds cannot declare output handles."));
            continue;
        }
        def.insert("outputs", definitions);
    }
    QJsonObject view;
    view.insert("color", color->text().trimmed());
    view.insert("width", 190);
    view.insert("height", 110);
    def.insert("view", view);
    QJsonObject params;
    const QSet<QString> parameterTypes = {QStringLiteral("boolean"), QStringLiteral("integer"),
        QStringLiteral("number"), QStringLiteral("string"), QStringLiteral("dtype"),
        QStringLiteral("json")};
    for (int row = 0; row < parameters->rowCount(); ++row) {
        const QString key = cellText(parameters, row, 0);
        const QString type = cellText(parameters, row, 1);
        if (key.isEmpty() || params.contains(key)) {
            formError->setText(tr("Parameter keys must be nonempty and unique.")); break;
        }
        if (!parameterTypes.contains(type)) {
            formError->setText(tr("Choose boolean, integer, number, string, dtype or JSON array.")); break;
        }
        QJsonObject parameter; parameter.insert("type", type);
        bool validDefault = false;
        const QJsonValue defaultValue = parseDefault(cellText(parameters, row, 2), type, &validDefault);
        if (!validDefault) { formError->setText(tr("Default value does not match the selected type.")); break; }
        parameter.insert("default", defaultValue);
        const QString minimum = cellText(parameters, row, 3);
        if (!minimum.isEmpty()) {
            bool ok = false; const double value = minimum.toDouble(&ok);
            if (!ok || !std::isfinite(value)) { formError->setText(tr("Minimum must be a finite number.")); break; }
            parameter.insert("minimum", value);
        }
        const QString choices = cellText(parameters, row, 4);
        if (!choices.isEmpty()) { QJsonArray array; for (const QString &choice : choices.split(',', Qt::SkipEmptyParts)) array.append(choice.trimmed()); parameter.insert("choices", array); }
        const QString position = cellText(parameters, row, 5);
        if (!position.isEmpty()) parameter.insert("position", position);
        params.insert(key, parameter);
    }
    if (!formError->text().isEmpty()) continue;
    def.insert("parameters", params);
    QJsonObject deps;
    for (int row = 0; row < dependencies->rowCount(); ++row) {
        const QString package = cellText(dependencies, row, 0), constraint = cellText(dependencies, row, 1);
        if (package.isEmpty() || constraint.isEmpty() || deps.contains(package)) {
            formError->setText(tr("Dependency IDs and constraints must be nonempty and unique.")); break;
        }
        deps.insert(package, constraint);
    }
    if (!formError->text().isEmpty()) continue;
    const QByteArray idBytes = id->text().trimmed().toUtf8();
    const QByteArray versionBytes = version->text().trimmed().toUtf8();
    const QByteArray definition = QJsonDocument(def).toJson(QJsonDocument::Compact);
    const QByteArray dependencyJson = QJsonDocument(deps).toJson(QJsonDocument::Compact);
    const QString luaText = lua->toPlainText();
    const QByteArray luaBytes = (luaText.trimmed().isEmpty() ? passThroughRule : luaText).toUtf8();
    char error[ErrorCapacity] = {};
    if (!nn_app_create_stereotype(application_.get(), idBytes.constData(), versionBytes.constData(),
            definition.constData(), luaBytes.constData(), dependencyJson.constData(), error, sizeof(error))) {
        const size_t line = nn_inference_error_line(error);
        luaError->setText(line ? tr("Lua error on line %1: %2").arg(line).arg(QString::fromUtf8(error))
                               : QString::fromUtf8(error));
        if (line) {
            QTextEdit::ExtraSelection selection;
            QTextCursor cursor(lua->document()->findBlockByNumber(int(line - 1)));
            if (cursor.isNull()) cursor = QTextCursor(lua->document()->lastBlock());
            selection.cursor = cursor;
            selection.format.setBackground(QColor("#ffe0df"));
            lua->setExtraSelections({selection});
        }
        continue;
    }
    refreshAll();
    statusBar()->showMessage(tr("Stereotype created; current project saved"), 6000);
    break;
    }
}
