// zip.workerScriptsPath = '/js/';

const bmf = new BMF();

var URL = window.URL || window.webkitURL;

var g_current_img = {
    name: "",
    upload_name: "",
    uploaded: false,
    md5:""
}

var body2 = ""

$(document).ready(function () {
    body2 = $('body').clone();

    $.ajaxSetup({
        cache: false
    });

    //从服务端获取风格类型
    getStyleList();

    //显示大图
    $("body").delegate('.img_upload', 'click', function(){  
        var _this = $(this);//将当前的pimg元素作为_this传入函数  
        imgShow("#outerdiv", "#innerdiv", "#bigimg", _this);  
        console.log('click img');
    });

    $('#file_upload').change(function () {
        $('#style_list').find('img').attr('hidden', true)
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

    $('#style_list').delegate('.style_name', 'click', function(){
        console.log( "select style:" + $(this).html());
        let style_id = $(this).parent().attr('style_id')
        if(g_current_img.uploaded == true)
            $(this).parent().find('img').attr('src', 'ico/loading.gif')
                .removeAttr('hidden')
                .attr('class', 'img_center')
            
        
        if(style_id != undefined){
            style_id = parseInt(  style_id  )
            beginTransfer(style_id);
        }
    })

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

        //获取md5
        bmf.md5(file, function (err, md5) {
            console.log("md5:" + md5); // 97027eb624f85892c69c4bcec8ab0f11
            uploadImg(md5);
        });
        
        
    });
}

function getStyleList(){
    $.ajax({
        url:"/api/get_styles_list",
        type: "POST",
        // data: formData,
        async: true,
        // cashe: false,
        contentType:false,
        processData:false,
        success:function (data) {
            console.log(data) 
            if(data.status == "ok" &&  data.num > 0){
                let parent = $('#style_list')
                let first = parent.children().first()
                let next = first.clone();
                next.removeAttr('hidden')
                // parent.children('li:gt(0)').remove()
                parent.children().remove()
                console.log(data.styles)
                for(i = 0; i < data.num; i++){
                    style = data.styles[i]
                    console.log(style)
                    next.find('li').html(style.name)
                    next.attr('style_id', style.id)
                    parent.append(next.clone())
                }
            }
            else{
                console.log("没有风格！")
            }
    　}, 
    　error: function (returndata) { 
    　　console.log("获取风格失败！")
    　}
    });

}


function uploadImg(md5) {
    console.log("upload " + new Date().getTime())
    $.ajax({
        url:"/api/is_file_exist",
        type:"POST",
        data: JSON.stringify({
            md5: md5
        }),
        contentType:false,
        success: function(data){
            if(data.status == 'ok' && data.file_exist == 'true'){
                console.log("file already exists!");
                g_current_img.upload_name = data.file_name;
                g_current_img.uploaded = true;
            }
            else{
                beginUpload(md5);
            }
        }
    })
}

function beginUpload(md5){
    var formData = new FormData($('#uploadForm')[0]);
    formData.set('md5', md5);
    $.ajax({
        url:"/api/upload_img", type: "POST",
        data: formData,
        contentType:false,
        processData:false,
        success:function (data) {
            console.log("/api/upload_img success!")
            console.log(data) 
            if(data.status == "ok"){
                g_current_img.upload_name = data.file_name;
                g_current_img.uploaded = true;
            }
    　}, 
    　error: function (returndata) { 
    　　console.log("上传失败！")
            g_current_img.upload_name = "";
            g_current_img.uploaded = false;
    　}
    });
}

function beginTransfer(style_id){
    let file_name = g_current_img.upload_name
    if(g_current_img.uploaded){
        $.ajax({
            url:"/api/begin_style_transfer",
            type: "POST",
            contentType: "application/json;charset=utf-8",
            data: JSON.stringify({"file_name": file_name, "style_id": style_id}),
            // dataType: "json",
            async: true,
            cashe: false,
            contentType:false,
            processData:false,
            success:function (data) {
                console.log("/api/begin_style_transfer success!")
                console.log(data) 
                if(data.status == "ok"){
                    let elem = $("#style_list").children(`[style_id=${data.style_id}]`)
                    elem.find('img').attr('src',  data.url );
                    elem.find('img').attr('class', 'img_upload')
                    elem.find('img').removeAttr('hidden')
                    //$("#style_list").children(`li[style_id=${data.style_id}]`).after(elem)
                    //$('#div_upload').append(elem);
                }
        　}, 
        　error: function (returndata) { 
        　　console.log("迁移失败！")
        　}
        });
    }
    else{
        console.log("Image is not uploaded!")
    }
}