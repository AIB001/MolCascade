/* A minimal DOM good enough to run the cascade builder outside a browser.
 *
 * The builder is a DOM application, so `node --check` proves almost nothing
 * about it.  No headless DOM is vendored here and none may be installed, so
 * this shim implements exactly the surface `cascade.js` touches: element
 * creation, text, attributes, datasets, listeners, and the two dialogs.  It is
 * a test fixture, not a browser -- layout, CSS and event bubbling are absent by
 * design, which is why the assertions built on it are about state, not pixels.
 */
"use strict";

function makeDocument(ids) {
  const registry = new Map();

  class ClassList {
    constructor(node) {
      this.node = node;
    }
    _tokens() {
      return (this.node.className || "").split(/\s+/).filter(Boolean);
    }
    _write(tokens) {
      this.node.className = tokens.join(" ");
    }
    contains(token) {
      return this._tokens().indexOf(token) >= 0;
    }
    add(token) {
      const tokens = this._tokens();
      if (tokens.indexOf(token) < 0) tokens.push(token);
      this._write(tokens);
    }
    remove(token) {
      this._write(this._tokens().filter((item) => item !== token));
    }
  }

  class Element {
    constructor(tag) {
      this.tagName = String(tag).toUpperCase();
      this.childNodes = [];
      this.parentNode = null;
      this.attributes = {};
      this.dataset = {};
      this.listeners = {};
      this.className = "";
      this.value = "";
      this.checked = false;
      this.disabled = false;
      this.hidden = false;
      this.tabIndex = -1;
      this.files = null;
      this._text = "";
      this.open = false;
      // Only custom properties are ever set from the builder, and only to carry
      // a stage group's accent colour, so recording them is enough.
      this.style = {
        properties: {},
        setProperty: (name, value) => {
          this.style.properties[name] = value;
        },
        getPropertyValue: (name) => this.style.properties[name] || "",
      };
    }

    get classList() {
      return new ClassList(this);
    }

    get textContent() {
      if (!this.childNodes.length) return this._text;
      return this.childNodes.map((child) => child.textContent).join("");
    }

    set textContent(value) {
      this.childNodes = [];
      this._text = value === null || value === undefined ? "" : String(value);
    }

    appendChild(child) {
      child.parentNode = this;
      this._text = "";
      this.childNodes.push(child);
      return child;
    }

    removeChild(child) {
      this.childNodes = this.childNodes.filter((item) => item !== child);
      child.parentNode = null;
      return child;
    }

    setAttribute(name, value) {
      this.attributes[name] = String(value);
      if (name === "class") this.className = String(value);
      if (name === "disabled") this.disabled = true;
      if (name === "hidden") this.hidden = true;
      if (name === "id") registry.set(String(value), this);
    }

    getAttribute(name) {
      return Object.prototype.hasOwnProperty.call(this.attributes, name)
        ? this.attributes[name]
        : null;
    }

    addEventListener(type, handler) {
      if (!this.listeners[type]) this.listeners[type] = [];
      this.listeners[type].push(handler);
    }

    removeEventListener(type, handler) {
      if (!this.listeners[type]) return;
      this.listeners[type] = this.listeners[type].filter((item) => item !== handler);
    }

    dispatch(type, extra) {
      const event = Object.assign(
        {
          type: type,
          target: this,
          currentTarget: this,
          preventDefault: function () {},
          stopPropagation: function () {},
        },
        extra || {}
      );
      for (const handler of (this.listeners[type] || []).slice()) handler(event);
      return event;
    }

    click() {
      if (this.disabled) throw new Error("clicked a disabled control: " + this.describe());
      this.dispatch("click");
    }

    focus() {
      document.activeElement = this;
    }

    showModal() {
      this.open = true;
    }

    close() {
      this.open = false;
    }

    describe() {
      return this.tagName + "." + (this.className || "-") + " “" + this.textContent.slice(0, 40) + "”";
    }

    walk(visit) {
      visit(this);
      for (const child of this.childNodes) child.walk(visit);
    }

    all(predicate) {
      const found = [];
      this.walk((node) => {
        if (predicate(node)) found.push(node);
      });
      return found;
    }

    querySelectorAll(selector) {
      return this.all(matcher(selector));
    }

    querySelector(selector) {
      return this.querySelectorAll(selector)[0] || null;
    }
  }

  function matcher(selector) {
    const attribute = /^\[([a-z-]+)="(.*)"\]$/.exec(selector);
    if (attribute) {
      const name = attribute[1];
      const wanted = attribute[2];
      const datasetKey = name.startsWith("data-")
        ? name
            .slice(5)
            .replace(/-([a-z])/g, (whole, letter) => letter.toUpperCase())
        : null;
      return (node) =>
        datasetKey ? node.dataset[datasetKey] === wanted : node.getAttribute(name) === wanted;
    }
    if (selector.startsWith(".")) {
      const wanted = selector.slice(1);
      return (node) => new ClassList(node).contains(wanted);
    }
    const tag = selector.toUpperCase();
    return (node) => node.tagName === tag;
  }

  const document = {
    activeElement: null,
    createElement: (tag) => new Element(tag),
    getElementById: (id) => registry.get(id) || null,
  };
  document.body = new Element("body");

  for (const id of ids) {
    const element = new Element(id === "builder-payload" ? "textarea" : "div");
    element.setAttribute("id", id);
    registry.set(id, element);
  }

  return { document: document, Element: Element };
}

module.exports = { makeDocument: makeDocument };
