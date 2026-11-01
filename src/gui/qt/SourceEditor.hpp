#ifndef NN_SOURCE_EDITOR_HPP
#define NN_SOURCE_EDITOR_HPP

#include <QColor>
#include <QFont>
#include <QPlainTextEdit>
#include <QRegularExpression>
#include <QSyntaxHighlighter>
#include <QTextCharFormat>

class SourceHighlighter final : public QSyntaxHighlighter {
public:
    SourceHighlighter(QTextDocument *document, const QString &language)
        : QSyntaxHighlighter(document), language_(language) {}

protected:
    void highlightBlock(const QString &text) override {
        QTextCharFormat keyword, string, comment;
        keyword.setForeground(QColor("#8050a0")); keyword.setFontWeight(QFont::Bold);
        string.setForeground(QColor("#287b50"));
        comment.setForeground(QColor("#777777"));
        const QString words = language_ == QStringLiteral("lua")
            ? QStringLiteral("and|break|do|else|elseif|end|false|for|function|if|in|local|nil|not|or|repeat|return|then|true|until|while")
            : QStringLiteral("and|as|assert|async|await|break|class|continue|def|del|elif|else|except|False|finally|for|from|if|import|in|is|lambda|None|not|or|pass|raise|return|True|try|while|with|yield");
        QRegularExpression keywords(QStringLiteral("\\b(?:%1)\\b").arg(words));
        auto matches = keywords.globalMatch(text);
        while (matches.hasNext()) { const auto match = matches.next(); setFormat(match.capturedStart(), match.capturedLength(), keyword); }
        // Scan strings and comments together so comment markers inside strings stay text.
        const QRegularExpression tokens(language_ == QStringLiteral("lua")
            ? QStringLiteral(R"(("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|--.*))")
            : QStringLiteral(R"(("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|#.*))"));
        matches = tokens.globalMatch(text);
        while (matches.hasNext()) {
            const auto match = matches.next();
            const QString token = match.captured();
            setFormat(match.capturedStart(), match.capturedLength(),
                      token.startsWith(QLatin1Char('#')) || token.startsWith(QStringLiteral("--")) ? comment : string);
        }
    }
private:
    QString language_;
};

class SourceEditor final : public QPlainTextEdit {
public:
    SourceEditor(const QString &language, QWidget *parent = nullptr) : QPlainTextEdit(parent) {
        new SourceHighlighter(document(), language);
        setTabStopDistance(fontMetrics().horizontalAdvance(QLatin1Char(' ')) * 4);
    }
};

#endif
