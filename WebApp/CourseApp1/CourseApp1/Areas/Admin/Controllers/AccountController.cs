using CourseApp1.Areas.Admin.Data;
using CourseApp1.services;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.Mvc;

namespace CourseApp1.Areas.Admin.Controllers
{
    public class AccountController : Controller
    {
        // GET: Admin/Account
        public ActionResult login()
        {
            return View();
        }
        [HttpPost]
        public ActionResult login(LoginModel loginfo)
        {
            var adminserv = new AdminServices();
            var islogged=adminserv.login(loginfo.Email, loginfo.Password);
            if (islogged)
            {
                return RedirectToAction("Index", "Default");
            }
            else
            {
                loginfo.message = "Email or Password icorrect";
                return View(loginfo);
            }
        }
    }
}