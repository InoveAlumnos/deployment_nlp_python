/*
Copyright (c) 2021 by Zeno Rocha (https://codepen.io/zenorocha/pen/eZxYOK)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

*/

// Variables globales:

var welcomeMessage = "¡Bienvenido!";
var welcome = document.querySelector("#welcome");
welcome.innerHTML = welcomeMessage;

// De tiempo. . .
var deviceTime = document.querySelector(".status-bar .time");
var messageTime = document.querySelectorAll(".message .time");
deviceTime.innerHTML = moment().format("h:mm");

// Del mensaje . . .

var form = document.querySelector(".conversation-compose");
var conversation = document.querySelector(".conversation-container");

// Formato de tiempo
setInterval(function () {
  deviceTime.innerHTML = moment().format("h:mm");
}, 1000);

// Este método escucha el botón submit y ejecuta la función newMessage
form.addEventListener("submit", newMessage);

// Esta función maneja los mensajes recibidos por el cliente y hace el request al chatbot
function newMessage(e) {
  var input = e.target.input;

  if (input.value) {
    //   input.value es el string que envió el cliente
    const message = input.value;
    const divmessage = buildMessage(input.value);
    // Esta es la url a la que le vamos a hacer el request:
    const url = "/v1/models/chatbot:predict";
    const payload = {"instances": [[message]]}
    conversation.appendChild(divmessage);
    animateMessage(divmessage);
    // El request se realiza con "fetch" y luego con la información
    // se construye la respuesta del bot en la pantalla
    fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
    })
      .then((response) => response.json())
      .then((data) => responseMessage(data));
  }
  // Limpiamos el input y hacemos scroll al final de la conversación:
  input.value = "";
  conversation.scrollTop = conversation.scrollHeight;

  e.preventDefault();
}

// Función que construye en la pantalla el mensaje de respuesta al bot
function responseMessage(data) {
  const text = data["predictions"][0];
  message = buildResponse(text);
  conversation.appendChild(message);
  animateMessage(message);
  conversation.scrollTop = conversation.scrollHeight;
}

// Esta función construye en la pantalla el mensaje del cliente
function buildMessage(text) {
  var element = document.createElement("div");

  element.classList.add("message", "sent");

  element.innerHTML =
    text +
    '<span class="metadata">' +
    '<span class="time">' +
    moment().format("h:mm A") +
    "</span>" +
    '<span class="tick tick-animation">' +
    '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="15" id="msg-dblcheck" x="2047" y="2061"><path d="M15.01 3.316l-.478-.372a.365.365 0 0 0-.51.063L8.666 9.88a.32.32 0 0 1-.484.032l-.358-.325a.32.32 0 0 0-.484.032l-.378.48a.418.418 0 0 0 .036.54l1.32 1.267a.32.32 0 0 0 .484-.034l6.272-8.048a.366.366 0 0 0-.064-.512zm-4.1 0l-.478-.372a.365.365 0 0 0-.51.063L4.566 9.88a.32.32 0 0 1-.484.032L1.892 7.77a.366.366 0 0 0-.516.005l-.423.433a.364.364 0 0 0 .006.514l3.255 3.185a.32.32 0 0 0 .484-.033l6.272-8.048a.365.365 0 0 0-.063-.51z" fill="#92a58c"/></svg>' +
    '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="15" id="msg-dblcheck-ack" x="2063" y="2076"><path d="M15.01 3.316l-.478-.372a.365.365 0 0 0-.51.063L8.666 9.88a.32.32 0 0 1-.484.032l-.358-.325a.32.32 0 0 0-.484.032l-.378.48a.418.418 0 0 0 .036.54l1.32 1.267a.32.32 0 0 0 .484-.034l6.272-8.048a.366.366 0 0 0-.064-.512zm-4.1 0l-.478-.372a.365.365 0 0 0-.51.063L4.566 9.88a.32.32 0 0 1-.484.032L1.892 7.77a.366.366 0 0 0-.516.005l-.423.433a.364.364 0 0 0 .006.514l3.255 3.185a.32.32 0 0 0 .484-.033l6.272-8.048a.365.365 0 0 0-.063-.51z" fill="#4fc3f7"/></svg>' +
    "</span>" +
    "</span>";

  return element;
}

// Esta función agrega el tilde azul y el timestamp del mensaje.
function animateMessage(message) {
  setTimeout(function () {
    var tick = message.querySelector(".tick");
    tick.classList.remove("tick-animation");
  }, 500);
}

// Esta función construye la respuesta recibiendo como parámetro el texto.
function buildResponse(text) {
  var element = document.createElement("div");
  element.classList.add("message", "received");
  element.innerHTML =
    text +
    '<span class="metadata">' +
    '<span class="time">' +
    moment().format("h:mm A");

  return element;
}

