// zip.workerScriptsPath = '/js/';

var URL = window.URL || window.webkitURL;

var saveBlob = (function () {
    var a = document.createElement("a");
    document.body.appendChild(a);
    a.style = "display: none";
    return function (blob, fileName) {
        var url = window.URL.createObjectURL(blob);
        a.href = url;
        a.download = fileName;
        a.click();
        window.URL.revokeObjectURL(url);
    };
}());



var get_suffix = function (f) {
    return (f.lastIndexOf('.') != -1) ? f.substring(f.lastIndexOf('.')) : '';
};

var generate_object_name = function (filename) {
    var d = new Date();
    return 'upload/' + d.getFullYear() + '-' + (d.getMonth() + 1) + '-' + d.getDate() + '/' + d
    .getTime() + Math.random() + get_suffix(filename);
};

$(document).ready(function () {
    $.ajaxSetup({
        cache: false
    });

    $('#file_upload').change(function () {
        file_change(this.files);
    });

    $(document).on('dragenter', function (e) {
        e.stopPropagation();
        e.preventDefault();
    });
    $(document).on('dragover', function (e) {
        e.stopPropagation();
        e.preventDefault();
    });
    $(document).on('drop', function (e) {
        e.preventDefault();
        file_change(e.originalEvent.dataTransfer.files);
    });

});

function file_change(files) {
    $.each(files, function (i, file) {
        if (!file.type.match(/^image\/*/)) return;

        let elem = $("<img  alt='image'>");
        elem.attr('src',  URL.createObjectURL(file) );
        elem.attr('class', 'img_upload')
        $('#div_upload').find('img').remove();
        $('#div_upload').append(elem);
       

        console.log('upload ' + new Date().getTime());

        /* jQuery 版 */
        uploadImg();
        
    });
}


function uploadImg() {
    console.log("upload " + new Date().getTime())
    var formData = new FormData($('#uploadForm')[0]);
    $.ajax({
        url:"/upload_img",
        type: "POST",
        data: formData,
        async: true,
        cashe: false,
        contentType:false,
        processData:false,
        success:function (data) {
            console.log(data) 
            if(data.status == "ok"){
                let elem = $("<img  alt='image'>");
                elem.attr('src',  data.url );
                elem.attr('class', 'img_upload')
                $('#div_upload').append(elem);
            }
    　}, 
    　error: function (returndata) { 
    　　console.log("上传失败！")
    　}
    })
}