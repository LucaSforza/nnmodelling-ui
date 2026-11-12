#include "ActivityLog.hpp"

#include <QAction>
#include <QApplication>
#include <QByteArray>
#include <QDialog>
#include <QEvent>
#include <QHash>
#include <QJsonDocument>
#include <QMenu>
#include <QPointer>
#include <QTimer>
#include <QToolBar>

namespace ActivityLog {
namespace {
NNActivityLog *g_log = nullptr;
int g_suppress = 0;
const QObject *g_shortcutAction = nullptr;
QHash<QByteArray, QByteArray> g_last;
QHash<QByteArray, QTimer *> g_timers;

// Notes which action a shortcut event reached so ui.action can report via "shortcut".
class ShortcutFilter final : public QObject {
public:
    explicit ShortcutFilter(QObject *parent) : QObject(parent) {}
    bool eventFilter(QObject *object, QEvent *event) override {
        if (event->type() == QEvent::Shortcut && qobject_cast<QAction *>(object)) {
            g_shortcutAction = object;
            QTimer::singleShot(0, this, [] { g_shortcutAction = nullptr; });
        }
        return false;
    }
};
ShortcutFilter *g_filter = nullptr;

const char *buttonName(QMessageBox::StandardButton button) {
    switch (button) {
    case QMessageBox::Ok: return "ok";
    case QMessageBox::Save: return "save";
    case QMessageBox::Discard: return "discard";
    case QMessageBox::Cancel: return "cancel";
    case QMessageBox::Close: return "close";
    case QMessageBox::Yes: return "yes";
    case QMessageBox::No: return "no";
    default: return "other";
    }
}

void schedule(QObject *context, const char *key, int ms, bool restart, std::function<void()> fn) {
    if (!g_log) return;
    const QByteArray name(key);
    QTimer *timer = g_timers.value(name, nullptr);
    if (!timer) {
        timer = new QTimer(qApp);
        timer->setSingleShot(true);
        g_timers.insert(name, timer);
    } else if (timer->isActive() && !restart) {
        return;
    }
    QPointer<QObject> guard(context);
    timer->disconnect();
    QObject::connect(timer, &QTimer::timeout, timer, [guard, fn = std::move(fn)] {
        if (guard && g_log) fn();
    });
    timer->start(ms);
}

QMessageBox::StandardButton message(QMessageBox::Icon icon, QWidget *parent, const QString &title,
    const QString &text, QMessageBox::StandardButtons buttons,
    QMessageBox::StandardButton defaultButton, const char *dialog) {
    if (!g_log)
        return icon == QMessageBox::Critical
            ? QMessageBox::critical(parent, title, text, buttons, defaultButton)
            : QMessageBox::warning(parent, title, text, buttons, defaultButton);
    dialogOpen(dialog);
    if (buttons == QMessageBox::Ok)
        record("error.shown", {{QStringLiteral("where"), QStringLiteral("dialog")},
                               {QStringLiteral("message"), redact(text)}});
    const QMessageBox::StandardButton result = icon == QMessageBox::Critical
        ? QMessageBox::critical(parent, title, text, buttons, defaultButton)
        : QMessageBox::warning(parent, title, text, buttons, defaultButton);
    dialogClose(dialog, buttonName(result));
    return result;
}
}

void setLog(NNActivityLog *log) {
    g_log = log;
    g_last.clear();
    g_shortcutAction = nullptr;
    qDeleteAll(g_timers);
    g_timers.clear();
    delete g_filter;
    g_filter = nullptr;
    if (log && qApp) {
        g_filter = new ShortcutFilter(qApp);
        qApp->installEventFilter(g_filter);
    }
}

bool enabled() { return g_log != nullptr; }

void record(const char *event, const QJsonObject &data, bool ok, const QString &error) {
    if (!g_log) return;
    const QByteArray json = QJsonDocument(data).toJson(QJsonDocument::Compact);
    const QByteArray errorText = error.toUtf8();
    nn_activity_emit(g_log, "ui", event, ok, ok ? nullptr : errorText.constData(),
                     json.constData());
}

void recordChanged(const char *event, const QJsonObject &data) {
    if (!g_log) return;
    const QByteArray json = QJsonDocument(data).toJson(QJsonDocument::Compact);
    const QByteArray key(event);
    if (g_last.value(key) == json && g_last.contains(key)) return;
    g_last.insert(key, json);
    record(event, data);
}

QString redact(const QString &text) {
    const QString home = QString::fromLocal8Bit(qgetenv("HOME"));
    if (home.size() < 2) return text;
    QString out = text;
    out.replace(home, QStringLiteral("~"));
    return out;
}

void coalesce(QObject *context, const char *key, std::function<void()> fn) {
    schedule(context, key, 0, false, std::move(fn));
}

void debounce(QObject *context, const char *key, int ms, std::function<void()> fn) {
    schedule(context, key, ms, true, std::move(fn));
}

void watchAction(QAction *action) {
    QObject::connect(action, &QAction::triggered, action, [action] {
        if (!g_log || g_suppress) return;
        QString via = QStringLiteral("other");
        if (g_shortcutAction == action) {
            via = QStringLiteral("shortcut");
        } else {
            for (const QObject *object : action->associatedObjects()) {
                if (qobject_cast<const QMenu *>(object)) { via = QStringLiteral("menu"); break; }
                if (qobject_cast<const QToolBar *>(object)) via = QStringLiteral("toolbar");
            }
        }
        QString name = action->objectName();
        if (name.isEmpty()) name = action->text();
        name.remove(QLatin1Char('&'));
        record("ui.action", {{QStringLiteral("action"), name}, {QStringLiteral("via"), via}});
    });
}

Suppress::Suppress() { ++g_suppress; }
Suppress::~Suppress() { --g_suppress; }

void dialogOpen(const char *dialog) {
    record("dialog.open", {{QStringLiteral("dialog"), QString::fromLatin1(dialog)}});
}

void dialogClose(const char *dialog, const char *result) {
    record("dialog.close", {{QStringLiteral("dialog"), QString::fromLatin1(dialog)},
                            {QStringLiteral("result"), QString::fromLatin1(result)}});
}

int exec(const char *dialog, QDialog &target) {
    dialogOpen(dialog);
    const int result = target.exec();
    dialogClose(dialog, result == QDialog::Accepted ? "accepted" : "rejected");
    return result;
}

QMessageBox::StandardButton warning(QWidget *parent, const QString &title, const QString &text,
    QMessageBox::StandardButtons buttons, QMessageBox::StandardButton defaultButton,
    const char *dialog) {
    return message(QMessageBox::Warning, parent, title, text, buttons, defaultButton, dialog);
}

QMessageBox::StandardButton critical(QWidget *parent, const QString &title, const QString &text,
    QMessageBox::StandardButtons buttons, QMessageBox::StandardButton defaultButton,
    const char *dialog) {
    return message(QMessageBox::Critical, parent, title, text, buttons, defaultButton, dialog);
}

QMessageBox::StandardButton question(const char *dialog, QWidget *parent, const QString &title,
    const QString &text, QMessageBox::StandardButtons buttons,
    QMessageBox::StandardButton defaultButton) {
    dialogOpen(dialog);
    const auto result = QMessageBox::question(parent, title, text, buttons, defaultButton);
    dialogClose(dialog, buttonName(result));
    return result;
}

QString existingDirectory(const char *dialog, QWidget *parent, const QString &caption,
    const QString &directory, QFileDialog::Options options) {
    dialogOpen(dialog);
    const QString path = QFileDialog::getExistingDirectory(parent, caption, directory, options);
    dialogClose(dialog, path.isEmpty() ? "rejected" : "accepted");
    return path;
}

QString saveFileName(const char *dialog, QWidget *parent, const QString &caption,
    const QString &directory) {
    dialogOpen(dialog);
    const QString path = QFileDialog::getSaveFileName(parent, caption, directory);
    dialogClose(dialog, path.isEmpty() ? "rejected" : "accepted");
    return path;
}

}
